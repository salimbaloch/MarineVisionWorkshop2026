#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m3_contours.py  —  Milestone 3: CONTOUR atoms (PiDiNet edges, mask-refereed)
+ relations + consolidated facts + chamfer QA.  ***FROZEN VERSION***

This is the finalized M3. With M1 and M2 it forms a frozen block sharing ONE
body definition:

  * BODY region atom  == m2_regions.ellipse_body_atom  (ellipse params +
    fit_error + DINOv3 feature_objectness/region_coherence). M3 no longer builds
    its own divergent blob body; route_topology still runs (its branch label is
    emitted as topology(...)), but the body atom itself is the M2 ellipse, so the
    consolidated facts carry the SAME body as M2 and M4. ridge_snap stays OFF
    (M2's QA decision: the B_dino ridge is the spine halo, not the body edge).

  * CONTOUR atoms  -- PiDiNet edge-probability on the full-res crop, the MASK
    referees every fragment (perimeter -> outer_contour_fragment, inside ->
    inner_contour_fragment, far outside -> discarded). Silhouette fallback for
    turbid crops. PiDiNet's real job in this block is the BODY BOUNDARY and
    FOREIGN-OBJECT (shell) edges.

SCOPE LOCK: M3 makes NO spine claims. Spines are entirely M4's concern, so
PiDiNet's silence in the spike annulus is never read as evidence here.

Cross-check emitted for ASP: contour_mask_agreement = how much of the mask
perimeter is corroborated by a PiDiNet outer fragment.

QA (--qa): chamfer( predicted boundary , GT polygon outline ), comparing the
PiDiNet outer boundary against the raw mask boundary and the silhouette.

Run on your machine. Deps: torch, numpy, cv2, PIL, scipy, matplotlib, core.py,
m2_regions.py, pidinet_edge.py + pidinet_pkg/ + table7_pidinet.pth (same folder).
"""
import os
import math
import shutil
import argparse

import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import core
from m2_regions import (route_topology, blob_region_atoms, shape_descriptors,
                        classify_form, ellipse_body_atom)
from pidinet_edge import PiDiNetEdger


# ============================================================ region atoms (via M2)
def region_atoms(mask_u8, support, mask_grid, crop_hw, probe, bimod_thr,
                 tok, grid, obj_proto, bg_proto, with_notches=False):
    """Route + build region atoms. FROZEN: the BODY is the M2 ellipse atom
    (m2_regions.ellipse_body_atom), so M3's consolidated facts carry the SAME
    body (ellipse params + fit_error + DINO objectness/coherence) as M2 and M4 --
    no divergent blob body. ridge_snap stays OFF (M2's QA decision). Optional
    concave notch sub-atoms behind with_notches (default off), mirroring M2.

    bdino_up is passed as None to ellipse_body_atom on purpose: with
    ridge_snap=False the builder never touches it, so M3 needs no B_dino map."""
    desc0 = shape_descriptors(mask_u8)
    if desc0 is None:
        return "none", []
    branch, info = route_topology(mask_u8, desc0, probe, bimod_thr)
    body = ellipse_body_atom(mask_u8, support, None, crop_hw, tok, grid,
                             mask_grid, obj_proto, bg_proto, ridge_snap=False)
    if body is None:
        return branch, []
    atoms = [body]
    if with_notches and branch == "blob":
        atoms += [a for a in blob_region_atoms(mask_u8, support, mask_grid, crop_hw)
                  if a["role"] == "concave_region"]
    return branch, atoms


# ============================================================ contour atoms (PiDiNet)
def _pca_orient_straightness(xs, ys):
    P = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float64)
    if len(P) < 2:
        return 0.0, 1.0
    w, v = np.linalg.eigh(np.cov(P.T))
    major = v[:, int(np.argmax(w))]
    orient = math.degrees(math.atan2(major[1], major[0])) % 180.0
    resid = math.sqrt(max(w.min(), 0.0))
    length = math.sqrt(max(w.max(), 0.0)) * 2 + 1
    return orient, resid / (length + 1e-6)        # ~0 straight, larger = curved


def _classify_by_mask(xs, ys, perim_band, interior, on_perim_min, inside_min):
    on_perim = float(perim_band[ys, xs].mean())
    inside = float(interior[ys, xs].mean())
    if on_perim >= on_perim_min:
        return "outer_contour_fragment", on_perim
    if inside >= inside_min:
        return "inner_contour_fragment", on_perim
    return None, on_perim                          # background edge -> mask vetoes it


def silhouette_contours(mask_u8, p):
    """Fallback path: trace the SILHOUETTE boundary itself (smoothed mask contour).
    Agrees with the mask boundary by construction, but is a smooth sub-grid curve
    and stays reliable when the crop is too turbid for real edges. Tagged
    source='silhouette' so ASP can prefer it on murky detections."""
    H, W = mask_u8.shape
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return []
    eps = p["silhouette_eps"] * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    m = np.zeros((H, W), np.uint8)
    cv2.drawContours(m, [approx], -1, 255, 2)
    xs, ys = approx[:, 0, 0], approx[:, 0, 1]
    return [dict(kind="contour", ctype="outer_contour_fragment", shape="curve_segment",
                 source="silhouette", length=int(cv2.arcLength(approx, True)),
                 centroid=(float(xs.mean()), float(ys.mean())),
                 bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                 endpoints=None, orient=None, edge_strength=100, support=88.0,
                 mask=m, primary=True, silhouette=True)]


def extract_contours(crop_rgb, mask_u8, edger, p):
    """PiDiNet edge prob -> threshold -> Hough straight + residual curves, each
    refereed by the mask. Plus corner_points on the perimeter. Returns
    (atoms, mask_boundary_pts, edges_obj). PiDiNet prob is the edge-strength."""
    H, W = mask_u8.shape
    interior = (mask_u8 > 0).astype(np.uint8)
    ker = np.ones((3, 3), np.uint8)
    er = cv2.erode(interior, ker, iterations=p["perim_px"])
    perim_band = (cv2.dilate(interior, ker, iterations=p["perim_px"]) - er) > 0
    obj_band = cv2.dilate(interior, ker, iterations=p["perim_px"] + 2) > 0  # referee region

    # ---- PiDiNet edge probability (replaces Canny) ----
    prob = edger.prob(crop_rgb)                      # HxW float in [0,1]
    gmag = prob                                      # PiDiNet prob IS the strength
    edges = (prob >= p["edge_thr"]).astype(np.uint8) * 255
    edges_obj = (edges > 0) & obj_band               # MASK REFEREES before anything
    edges_obj_u8 = (edges_obj.astype(np.uint8)) * 255

    # CLAHE gray is still used only for corner detection
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=(8, 8)).apply(gray)

    atoms = []

    # ---- straight segments: Hough on the PiDiNet edge map ----
    line_mask = np.zeros((H, W), np.uint8)
    lines = cv2.HoughLinesP(edges_obj_u8, 1, np.pi / 180, p["hough_thr"],
                            minLineLength=p["hough_min_len"], maxLineGap=p["hough_gap"])
    if lines is not None:
        for ln in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in ln]
            m = np.zeros((H, W), np.uint8)
            cv2.line(m, (x1, y1), (x2, y2), 255, 2)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
            ys, xs = np.where(m > 0)
            ctype, on_perim = _classify_by_mask(xs, ys, perim_band, interior,
                                                p["on_perim_min"], p["inside_min"])
            if ctype is None:
                continue
            length = float(math.hypot(x2 - x1, y2 - y1))
            orient = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            strength = float(gmag[ys, xs].mean())
            support = float(np.clip(0.55 * strength + 0.45 * on_perim, 0, 1) * 100)
            atoms.append(dict(kind="contour", source="image_edge", ctype=ctype,
                              shape="edge_segment", length=int(length),
                              centroid=(float(xs.mean()), float(ys.mean())),
                              bbox=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                              endpoints=((x1, y1), (x2, y2)), orient=orient,
                              edge_strength=int(strength * 100), support=support, mask=m))

    # ---- residual CURVES: edges not captured by Hough ----
    residual = (edges_obj & (cv2.dilate(line_mask, ker, 1) == 0)).astype(np.uint8)
    n, lbl = cv2.connectedComponents(residual)
    cand = []
    for i in range(1, n):
        ys, xs = np.where(lbl == i)
        if len(xs) < p["min_len"]:
            continue
        ctype, on_perim = _classify_by_mask(xs, ys, perim_band, interior,
                                            p["on_perim_min"], p["inside_min"])
        if ctype is None:
            continue
        orient, straight = _pca_orient_straightness(xs, ys)
        shape = "edge_segment" if straight < p["straight_thr"] else "curve_segment"
        strength = float(gmag[ys, xs].mean())
        support = float(np.clip(0.55 * strength + 0.45 * on_perim, 0, 1) * 100)
        if support < p["min_support"]:
            continue
        m = np.zeros((H, W), np.uint8); m[ys, xs] = 255
        cand.append(dict(kind="contour", source="image_edge", ctype=ctype, shape=shape,
                         length=int(len(xs)),
                         centroid=(float(xs.mean()), float(ys.mean())),
                         bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                         endpoints=None, orient=orient, edge_strength=int(strength * 100),
                         support=support, mask=m))
    cand.sort(key=lambda a: -a["support"])
    atoms += cand[:p["max_frag"]]

    # ---- corner_points on the perimeter (spine tips for urchin) ----
    cor = cv2.goodFeaturesToTrack(clahe, p["max_corners"], 0.04,
                                  int(0.05 * max(H, W) + 2))
    if cor is not None:
        for px, py in cor.reshape(-1, 2):
            cx, cy = int(px), int(py)
            if 0 <= cy < H and 0 <= cx < W and perim_band[cy, cx]:
                m = np.zeros((H, W), np.uint8)
                cv2.circle(m, (cx, cy), max(2, p["perim_px"]), 255, -1)
                atoms.append(dict(kind="contour", source="image_edge", ctype="corner_point",
                                  shape="corner", length=1, centroid=(float(cx), float(cy)),
                                  bbox=(cx-2, cy-2, cx+2, cy+2), endpoints=None,
                                  orient=None, edge_strength=int(gmag[cy, cx] * 100),
                                  support=70.0, mask=m))

    # ---- mask boundary points (for agreement + chamfer only) ----
    cnts, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_boundary_pts = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        mask_boundary_pts = np.stack([c[:, 0, 0], c[:, 0, 1]], 1)
    return atoms, mask_boundary_pts, edges_obj


def contour_mask_agreement(atoms, mask_boundary_pts, tau=4):
    """Fraction of mask-perimeter pixels within tau px of a PiDiNet OUTER fragment."""
    if mask_boundary_pts is None:
        return 0.0
    outer = [a for a in atoms if a["ctype"] == "outer_contour_fragment"
             and not a.get("primary") and a.get("source") == "image_edge"]
    if not outer:
        return 0.0
    pts = np.vstack([np.stack(np.where(a["mask"] > 0)[::-1], 1) for a in outer])
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    d, _ = tree.query(mask_boundary_pts, k=1)
    return float((d <= tau).mean())


# ============================================================ relations
def _adiff(a, b):
    if a is None or b is None:
        return None
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _touch(ma, mb, k):
    d = cv2.dilate(ma, np.ones((3, 3), np.uint8), iterations=k)
    return bool(np.logical_and(d > 0, mb > 0).any())


def build_relations(regions, contours, proposal_box, p):
    """region<->region: adjacency/near/directional/parallel/concentric/encloses.
    region<->contour: near_contour/supports_contour. contour<->contour: collinear.

    NOTE (frozen): the body region is now the M2 ELLIPSE, so supports_contour
    tests fragments against the ellipse perimeter. The ellipse hugs the support
    mask (fit_error ~0.06 in M2 QA), so nearly all outer fragments still fire;
    where they don't, the real edge and the ellipse genuinely disagree there --
    a useful fact for ASP, not a bug. Contour fragments themselves are still
    refereed against the support mask in extract_contours, unchanged."""
    rel = []
    px1, py1, px2, py2 = proposal_box
    near_d = p["near_frac"] * (math.hypot(px2-px1, py2-py1) + 1e-6)
    R = regions
    for i in range(len(R)):
        for j in range(len(R)):
            if i == j:
                continue
            A, B = R[i], R[j]
            (ax, ay), (bx, by) = A["centroid"], B["centroid"]
            if i < j:
                if A.get("mask") is not None and B.get("mask") is not None \
                        and _touch(A["mask"], B["mask"], p["touch_px"]):
                    rel += [("adjacent", A["id"], B["id"])]
                if math.hypot(ax-bx, ay-by) < near_d:
                    rel.append(("near", A["id"], B["id"]))
                da = _adiff(A.get("orient"), B.get("orient"))
                if da is not None and da < 15:
                    rel.append(("parallel", A["id"], B["id"]))
                if math.hypot(ax-bx, ay-by) < 0.1 * near_d:
                    rel.append(("concentric", A["id"], B["id"]))
            if (A["bbox"][0] <= B["bbox"][0] and A["bbox"][1] <= B["bbox"][1]
                    and A["bbox"][2] >= B["bbox"][2] and A["bbox"][3] >= B["bbox"][3]
                    and A["id"] != B["id"]):
                rel.append(("encloses", A["id"], B["id"]))
            if bx - ax > p["dir_px"]:
                rel.append(("left_of", A["id"], B["id"]))
            if by - ay > p["dir_px"]:
                rel.append(("above", A["id"], B["id"]))
    for A in regions:
        if A.get("mask") is None:
            continue
        band = cv2.dilate(A["mask"], np.ones((3, 3), np.uint8), iterations=p["touch_px"])
        perim = cv2.dilate(A["mask"], np.ones((3, 3), np.uint8), iterations=p["perim_px"]) - \
            cv2.erode(A["mask"], np.ones((3, 3), np.uint8), iterations=p["perim_px"])
        for C in contours:
            if np.logical_and(band > 0, C["mask"] > 0).any():
                rel.append(("near_contour", A["id"], C["id"]))
            frac = np.logical_and(perim > 0, C["mask"] > 0).sum() / (float((C["mask"] > 0).sum()) + 1e-6)
            if frac > 0.4:
                rel.append(("supports_contour", A["id"], C["id"]))
    segs = [c for c in contours if c.get("endpoints") and c["shape"] == "edge_segment"]
    for i in range(len(segs)):
        for j in range(i+1, len(segs)):
            A, B = segs[i], segs[j]
            if _adiff(A["orient"], B["orient"]) is None or _adiff(A["orient"], B["orient"]) > 10:
                continue
            (ax1, ay1), (ax2, ay2) = A["endpoints"]
            (bx1c, by1c), _ = B["endpoints"]
            dx, dy = ax2-ax1, ay2-ay1
            L = math.hypot(dx, dy) + 1e-6
            perp = abs((bx1c-ax1)*(-dy) + (by1c-ay1)*dx) / L
            if perp < p["collinear_px"]:
                rel.append(("collinear", A["id"], B["id"]))
    return rel


# ============================================================ chamfer QA
def boundary_pixels_from_mask(mask_u8):
    g = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ys, xs = np.where(g > 0)
    return np.stack([xs, ys], 1) if len(xs) else None


def poly_outline_pixels(seg, H, W, ox, oy):
    if not isinstance(seg, list):
        return None
    canvas = np.zeros((H, W), np.uint8)
    for ring in seg:
        pts = np.array(ring, np.int32).reshape(-1, 2).copy()
        pts[:, 0] -= ox; pts[:, 1] -= oy
        cv2.polylines(canvas, [pts], True, 255, 1)
    ys, xs = np.where(canvas > 0)
    return np.stack([xs, ys], 1) if len(xs) else None


def chamfer(A, B):
    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return float("nan")
    from scipy.spatial import cKDTree
    da, _ = cKDTree(B).query(A, k=1)
    db, _ = cKDTree(A).query(B, k=1)
    return float((da.mean() + db.mean()) / 2)


# ============================================================ facts
def write_facts(path, det_box_crop, branch, regions, contours, relations, mq,
                agreement, oid="o1", cls="object"):
    L = ["% M3 consolidated facts (FROZEN): region (M2 ellipse body) + contour",
         "% (PiDiNet, mask-refereed) + relations.",
         "% directional relations in image-axis frame (smaller x=left, smaller y=up).",
         "% M3 makes NO spine claims; spines are M4's concern.", ""]
    bx = [int(round(v)) for v in det_box_crop]
    L += [f"detection({oid}).", f"class({oid},{cls}).",
          f"proposal_box({oid},box({bx[0]},{bx[1]},{bx[2]},{bx[3]})).",
          f"topology({oid},{branch})."]
    if mq is not None:
        L += [f"mask_contrast({oid},{int(round(mq['contrast']))}).",
              f"mask_frac({oid},{int(round(mq['mask_frac']*100))}).",
              f"boundary_sharpness({oid},{int(round(mq['boundary_sharpness']))}).",
              f"contour_mask_agreement({oid},{int(round(agreement*100))})."]
    # ---- region atoms (body == M2 ellipse atom; richer facts) ----
    for a in regions:
        cx, cy = [int(round(v)) for v in a["centroid"]]
        bb = [int(round(v)) for v in a["bbox"]]
        L += ["", f"atom({a['id']}).", f"belongs_to_detection({a['id']},{oid}).",
              f"atom_kind({a['id']},geometric_form).", f"atom_type({a['id']},{a['role']}).",
              f"shape_class({a['id']},{a['form']}).", f"centroid({a['id']},point({cx},{cy})).",
              f"atom_box({a['id']},box({bb[0]},{bb[1]},{bb[2]},{bb[3]})).",
              f"support({a['id']},{a['support']}).", f"inside({a['id']},{oid})."]
        # ellipse parameters + fit quality (Step 2, shared with M2/M4)
        if a.get("ellipse") is not None:
            ecx, ecy, ed1, ed2, eang = a["ellipse"]
            L += [f"ellipse({a['id']},point({int(round(ecx))},{int(round(ecy))}),"
                  f"axes({int(round(ed1))},{int(round(ed2))}),angle({int(round(eang))})).",
                  f"fit_error({a['id']},{int(round(a['fit_error']*100))})."]
        # Step-1 DINOv3 semantics on the atom
        if a.get("objness") is not None:
            L.append(f"feature_objectness({a['id']},{int(round(a['objness']))}).")
        if a.get("coher") is not None:
            L.append(f"region_coherence({a['id']},{int(round(a['coher']))}).")
        d = a.get("desc")
        if d:
            L += [f"area({a['id']},{int(round(d['area']))}).",
                  f"elongation({a['id']},{int(round(d['elongation']*100))}).",
                  f"circularity({a['id']},{int(round(d['circularity']*100))}).",
                  f"solidity({a['id']},{int(round(d['solidity']*100))}).",
                  f"convexity_defects({a['id']},{d['n_defects']})."]
        if "defect_depth" in a:
            L.append(f"defect_depth({a['id']},{a['defect_depth']}).")
    # ---- contour atoms (PiDiNet boundary + foreign-object edges) ----
    for a in contours:
        cid = a["id"]
        cx, cy = [int(round(v)) for v in a["centroid"]]
        bb = [int(round(v)) for v in a["bbox"]]
        L += ["", f"atom({cid}).", f"belongs_to_detection({cid},{oid}).",
              f"atom_kind({cid},contour).", f"atom_type({cid},{a['ctype']}).",
              f"contour_source({cid},{a.get('source','image_edge')}).",
              f"contour_shape({cid},{a['shape']}).", f"length({cid},{a['length']}).",
              f"centroid({cid},point({cx},{cy})).",
              f"atom_box({cid},box({bb[0]},{bb[1]},{bb[2]},{bb[3]})).",
              f"edge_strength({cid},{a['edge_strength']}).",
              f"support({cid},{int(round(a['support']))}).", f"inside({cid},{oid})."]
        if a.get("orient") is not None:
            L.append(f"orientation({cid},{int(round(a['orient']))}).")
    L.append("")
    for r, x, y in relations:
        L.append(f"{r}({x},{y}).")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ============================================================ visualize
def visualize(crop_rgb, det_box_crop, mask_u8, regions, contours, out_path, title):
    H, W = crop_rgb.shape[:2]
    col = {"outer_contour_fragment": "yellow", "inner_contour_fragment": "cyan",
           "corner_point": "red"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    bx1, by1, bx2, by2 = det_box_crop

    ax[0].imshow(crop_rgb)
    ax[0].add_patch(Rectangle((bx1, by1), bx2-bx1, by2-by1, fill=False, ec="lime", lw=2))
    ov = np.zeros((H, W, 4)); ov[mask_u8 > 0] = [1, 1, 1, 0.25]; ax[0].imshow(ov)
    # ellipse body outline (yellow) over the support mask
    for a in regions:
        if a.get("form") == "ellipse" and a.get("mask") is not None:
            ax[0].contour(a["mask"].astype(float), levels=[127], colors="yellow", linewidths=1.4)
    ax[0].set_title("crop + DETR box + support mask + ellipse body")

    ax[1].imshow(crop_rgb)
    for a in contours:
        ys, xs = np.where(a["mask"] > 0)
        c = col.get(a["ctype"], "lime")
        if a["shape"] == "edge_segment":
            c = "lime"
        ax[1].scatter(xs, ys, s=2, c=c, marker="s")
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=c, label=t.replace("_contour_fragment", ""))
               for t, c in col.items()]
    handles.append(plt.Line2D([0], [0], marker="s", ls="", color="lime", label="straight (hough)"))
    ax[1].legend(handles=handles, fontsize=7, loc="lower right")
    ax[1].set_title("contour atoms (PiDiNet, mask-refereed)")

    ax[2].imshow(crop_rgb)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
        ax[2].plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], "w--", lw=1.2, label="mask boundary")
    for a in contours:
        if a.get("source") == "silhouette":
            ys, xs = np.where(a["mask"] > 0); ax[2].scatter(xs, ys, s=3, c="orange", marker="s")
    for a in contours:
        if a["ctype"] == "outer_contour_fragment" and not a.get("primary") and a.get("source") == "image_edge":
            ys, xs = np.where(a["mask"] > 0); ax[2].scatter(xs, ys, s=4, c="yellow", marker="s")
    h2 = [plt.Line2D([0], [0], ls="--", color="w", label="mask boundary"),
          plt.Line2D([0], [0], marker="s", ls="", color="yellow", label="PiDiNet outer"),
          plt.Line2D([0], [0], marker="s", ls="", color="orange", label="silhouette")]
    ax[2].legend(handles=h2, fontsize=7, loc="lower right")
    ax[2].set_title("boundaries: mask (white) / PiDiNet (yellow) / silhouette (orange)")

    for a in ax:
        a.set_xlim(0, W); a.set_ylim(H, 0); a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--weights-kind", choices=["auto", "pretrained", "deim"], default="deim")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--pred-json", "--pred", dest="pred_json", required=True)
    ap.add_argument("--gt-json", "--gt", dest="gt_json", required=True)
    ap.add_argument("--img-root", "--images", dest="img_root", required=True)
    ap.add_argument("--class-name", default="animal_urchin")
    ap.add_argument("--box-format", choices=["xywh", "xyxy"], default="xywh")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--seed-frac", type=float, default=0.5)
    ap.add_argument("--sup-thr-frac", type=float, default=0.55)
    ap.add_argument("--bimod-thr", type=float, default=2.2)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--min-grid", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--notches", action="store_true",
                    help="also emit concave notch sub-atoms (blob branch); off by default")
    # PiDiNet edge params (replace Canny)
    ap.add_argument("--pidinet-weights", default="table7_pidinet.pth")
    ap.add_argument("--pidinet-device", default=None)
    ap.add_argument("--edge-thr", type=float, default=0.4,
                    help="PiDiNet probability threshold to binarize edges")
    ap.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE for corner detection")
    ap.add_argument("--min-len", type=int, default=12)
    ap.add_argument("--straight-thr", type=float, default=0.08)
    ap.add_argument("--perim-px", type=int, default=3)
    ap.add_argument("--max-corners", type=int, default=12)
    ap.add_argument("--on-perim-min", type=float, default=0.5)
    ap.add_argument("--inside-min", type=float, default=0.6)
    ap.add_argument("--min-support", type=float, default=45)
    ap.add_argument("--max-frag", type=int, default=25)
    ap.add_argument("--hough-thr", type=int, default=20)
    ap.add_argument("--hough-min-len", type=int, default=18)
    ap.add_argument("--hough-gap", type=int, default=4)
    ap.add_argument("--silhouette-eps", type=float, default=0.01)
    # relation params
    ap.add_argument("--near-frac", type=float, default=0.10)
    ap.add_argument("--touch-px", type=int, default=3)
    ap.add_argument("--dir-px", type=int, default=6)
    ap.add_argument("--collinear-px", type=int, default=4)
    ap.add_argument("--qa", action="store_true", help="chamfer QA vs polygon; no png/lp")
    ap.add_argument("--max-dets", type=int, default=40)
    ap.add_argument("--out-dir", default="./m3_out")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.fresh and os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    id2file, name2cat, cat2name, gt_seg = core.load_gt_index(args.gt_json)
    if args.class_name not in name2cat:
        raise SystemExit(f"'{args.class_name}' not in {list(name2cat)[:12]}...")
    target = name2cat[args.class_name]
    print(f"Target '{args.class_name}' -> id={target}")
    import json
    id2hw = {im["id"]: (im["height"], im["width"]) for im in json.load(open(args.gt_json))["images"]}

    model = core.load_dinov3(args.repo_root, args.weights, args.weights_kind,
                             prefer_ema=not args.no_ema, device=device)
    edger = PiDiNetEdger(args.pidinet_weights, device=args.pidinet_device or device)
    by_img = core.load_predictions(args.pred_json, args.box_format)

    cparams = dict(edge_thr=args.edge_thr, clahe_clip=args.clahe_clip,
                   min_len=args.min_len, straight_thr=args.straight_thr, perim_px=args.perim_px,
                   max_corners=args.max_corners, on_perim_min=args.on_perim_min,
                   inside_min=args.inside_min, min_support=args.min_support, max_frag=args.max_frag,
                   hough_thr=args.hough_thr, hough_min_len=args.hough_min_len, hough_gap=args.hough_gap,
                   silhouette_eps=args.silhouette_eps)
    rparams = dict(near_frac=args.near_frac, touch_px=args.touch_px, dir_px=args.dir_px,
                   perim_px=args.perim_px, collinear_px=args.collinear_px)

    qa = []
    done = 0
    for image_id, dets in by_img.items():
        dets = [d for d in dets if d["cat"] == target and d["score"] >= args.score_thr]
        if not dets:
            continue
        fname = id2file.get(image_id)
        ipath = os.path.join(args.img_root, fname) if fname else None
        if not ipath or not os.path.exists(ipath):
            continue
        image_rgb = np.asarray(Image.open(ipath).convert("RGB"))
        gts = [g for g in gt_seg.get(image_id, []) if g["cat"] == target] if args.qa else []
        used = [False] * len(gts)
        for k, d in enumerate(dets):
            crop_rgb, (ox1, oy1, _, _) = core.crop_with_margin(image_rgb, d["box"], args.margin)
            if crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 6:
                continue
            ch, cw = crop_rgb.shape[:2]
            dbx = (d["box"][0]-ox1, d["box"][1]-oy1, d["box"][2]-ox1, d["box"][3]-oy1)
            ten, mhw = core.to_model_input(crop_rgb, args.min_grid, args.max_side, device)
            tok, grid = core.extract_tokens(model, ten, args.layer)
            support, gate = core.compute_support(tok, grid, mhw, dbx, (ch, cw), args.seed_frac)
            mask_grid = core.clean_mask(support, args.sup_thr_frac)
            stem = f"{os.path.splitext(os.path.basename(fname))[0]}_det{k}"
            if mask_grid is None:
                continue
            probe = core.feature_variance_probe(tok, grid, mask_grid)
            mq = core.mask_quality(support, mask_grid)
            mask_u8 = (core.upsample_mask(mask_grid, (ch, cw)).astype(np.uint8)) * 255

            # FROZEN: body region atom == M2 ellipse (needs object/bg prototypes)
            obj_proto, bg_proto = core.object_bg_prototypes(
                tok, grid, mhw, dbx, (ch, cw), args.seed_frac)
            branch, regions = region_atoms(mask_u8, support, mask_grid, (ch, cw),
                                           probe, args.bimod_thr, tok, grid,
                                           obj_proto, bg_proto, with_notches=args.notches)
            if not regions:
                continue
            contours, mask_bpts, _ = extract_contours(crop_rgb, mask_u8, edger, cparams)
            contours += silhouette_contours(mask_u8, cparams)
            for ci, a in enumerate(contours, 1):
                a["id"] = f"c{ci}"
            agreement = contour_mask_agreement(contours, mask_bpts)
            relations = build_relations(regions, contours, dbx, rparams)

            if args.qa:
                bj, best = -1, 0.1
                for j, g in enumerate(gts):
                    if used[j]:
                        continue
                    v = _bbox_iou(d["box"], g["box"])
                    if v >= best:
                        best, bj = v, j
                if bj < 0:
                    continue
                used[bj] = True
                gpts = poly_outline_pixels(gts[bj]["seg"], ch, cw, ox1, oy1)
                mask_b = boundary_pixels_from_mask(mask_u8)
                outer_ie = [a for a in contours if a["ctype"] == "outer_contour_fragment"
                            and not a.get("primary") and a.get("source") == "image_edge"]
                outer_sil = [a for a in contours if a.get("source") == "silhouette"]
                cpts = np.vstack([np.stack(np.where(a["mask"] > 0)[::-1], 1) for a in outer_ie]) \
                    if outer_ie else None
                spts = np.vstack([np.stack(np.where(a["mask"] > 0)[::-1], 1) for a in outer_sil]) \
                    if outer_sil else None
                qa.append(dict(ch_mask=chamfer(mask_b, gpts), ch_pidi=chamfer(cpts, gpts),
                               ch_sil=chamfer(spts, gpts), agreement=agreement,
                               n_outer=len(outer_ie), n_contour=len(contours), n_rel=len(relations)))
            else:
                write_facts(os.path.join(args.out_dir, stem + ".lp"), dbx, branch,
                            regions, contours, relations, mq, agreement, cls=args.class_name)
                visualize(crop_rgb, dbx, mask_u8, regions, contours,
                          os.path.join(args.out_dir, stem + ".png"),
                          title=f"{fname} det#{k} s={d['score']:.2f} branch={branch} "
                                f"contours={len(contours)} rels={len(relations)} "
                                f"agree={agreement:.2f}")
            done += 1
            if done >= args.max_dets:
                break
        if done >= args.max_dets:
            break

    if args.qa:
        qa_report(qa)
    else:
        print(f"\nWrote {done} detections to {args.out_dir}")


def qa_report(rows):
    if not rows:
        print("No matched detections."); return
    a = lambda k: np.array([r[k] for r in rows], float)
    cm = a("ch_mask"); cc = a("ch_pidi"); cs = a("ch_sil")
    vm = cm[~np.isnan(cm)]; vc = cc[~np.isnan(cc)]; vs = cs[~np.isnan(cs)]
    print(f"\nM3 CONTOUR QA  ({len(rows)} matched detections) ")
    print(f"  contour atoms / det     : mean {a('n_contour').mean():.1f}")
    print(f"  PiDiNet outer / det     : mean {a('n_outer').mean():.1f}")
    print(f"  relations / det         : mean {a('n_rel').mean():.1f}")
    print(f"  contour-mask agreement  : mean {a('agreement').mean():.2f}  (PiDiNet only)")
    print(f"\n  chamfer to GT polygon outline (px, lower = better):")
    print(f"    mask boundary      : mean {vm.mean():.2f}  median {np.median(vm):.2f}  (n={vm.size})")
    print(f"    PiDiNet edge       : mean {vc.mean():.2f}  median {np.median(vc):.2f}  (n={vc.size})")
    print(f"    silhouette         : mean {vs.mean():.2f}  median {np.median(vs):.2f}  (n={vs.size})")
    bM = ~np.isnan(cm) & ~np.isnan(cc)
    bS = ~np.isnan(cm) & ~np.isnan(cs)
    if bM.sum():
        print(f"\n  PiDiNet beats mask boundary on {100*(cc[bM] < cm[bM]).mean():.0f}% of dets")
    if bS.sum():
        print(f"  silhouette beats mask boundary on {100*(cs[bS] < cm[bS]).mean():.0f}% of dets")
    print(f"\n  note: GT polygon is the coarse body outline, so the BODY chamfer is")
    print(f"  capped by GT coarseness; the spikes PiDiNet adds are not in the GT.")


if __name__ == "__main__":
    main()