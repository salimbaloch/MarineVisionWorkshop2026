#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decompose.py  —  the LEAN urchin clean run (spec stages 1-2-3), SPINES SKIPPED.
Produces one self-contained .lp per detection for the Stage-4 (ASP) layer + a
debug panel.

CORRECTED BOUNDARY MODEL
------------------------
The body SILHOUETTE is the boundary. It is the semantic (DINOv3 support) edge and
is reliable even in turbid water where no hard edge exists. PiDiNet does NOT
define the boundary; it corroborates parts of it where a hard edge is present,
and otherwise marks INTERIOR structure (foreign-object rims, clefts).

Therefore:
  * boundary_primitive = the silhouette outline, ALWAYS emitted, carrying
    semantic_support (high) and edge_confidence (PiDiNet agreement along it,
    which may be low -- and that is honest).
  * PiDiNet fragments are classified by LOCATION:
      - OUTER (on the perimeter band)  -> corroborating boundary primitives.
      - INTERIOR (inside the body)     -> kept as geometric primitives tagged
        location(g,interior); NOT boundary. (Foreign-object rims, internal edges.)

Spec stages:
  STAGE 1  SEMANTIC REGIONS    body region + foreign-object regions (M1/M2 + DINO)
  STAGE 2  GEOMETRIC PRIMITIVES M2 ellipse + PiDiNet edge fragments (skeletonized,
                                each tagged location outer/interior)
  STAGE 3  BOUNDARY PRIMITIVES  silhouette (always) + OUTER PiDiNet fragments that
                                clear the edge gate; each carries edge_confidence
                                and semantic_support.

PiDiNet's role here, plainly: corroborate the body boundary (outer fragments) and
detect foreign objects (Stage-1 features). SPINES are NOT attempted (deferred).

Run on your machine. Deps: torch, numpy, cv2, PIL, scipy, matplotlib,
(skimage for skeletonize, optional), core.py, m2_regions.py, m4_decompose.py,
pidinet_edge.py + pidinet_pkg/ + table7_pidinet.pth.
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
from m2_regions import (ellipse_body_atom, ellipse_to_mask, shape_descriptors,
                        _ellipse_perim_band)
from m4_decompose import annulus_zone, shell_atoms        # foreign-object detector


# ============================================================ skeletonize
def _skeletonize(binmask):
    """Thin a boolean edge mask to 1px. skimage preferred; cv2.ximgproc fallback;
    else return the mask unchanged."""
    b = binmask > 0
    try:
        from skimage.morphology import skeletonize
        return skeletonize(b)
    except Exception:
        pass
    try:
        import cv2.ximgproc as xip
        return xip.thinning((b.astype(np.uint8)) * 255) > 0
    except Exception:
        return b


# ============================================================ STAGE 2: geometric primitives
def _fit_fragment(xs, ys, straight_thr):
    """PCA line/curve fit to a connected edge fragment. Returns type, orientation,
    length, endpoints, bbox, fit_error (straightness residual; 0 = straight)."""
    P = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float64)
    w, v = np.linalg.eigh(np.cov(P.T))
    major = v[:, int(np.argmax(w))]
    orient = math.degrees(math.atan2(major[1], major[0])) % 180.0
    resid = math.sqrt(max(w.min(), 0.0))
    length = math.sqrt(max(w.max(), 0.0)) * 2 + 1
    straight = resid / (length + 1e-6)
    ptype = "line_segment" if straight < straight_thr else "curve_segment"
    t = P @ major
    i1, i2 = int(np.argmin(t)), int(np.argmax(t))
    endpoints = ((int(xs[i1]), int(ys[i1])), (int(xs[i2]), int(ys[i2])))
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return dict(ptype=ptype, orient=orient, length=float(length),
                endpoints=endpoints, bbox=bbox, fit_error=float(straight))


def geometric_primitives(prob, mask_u8, p):
    """PiDiNet prob -> threshold -> skeletonize -> CC -> fit line/curve.
    Restricted to an object band so pure-seabed edges are dropped. edge_confidence
    = mean PiDiNet prob along the fragment (0-100). Returns a list (no ellipse)."""
    H, W = mask_u8.shape
    ker = np.ones((3, 3), np.uint8)
    obj_band = cv2.dilate((mask_u8 > 0).astype(np.uint8), ker,
                          iterations=p["obj_band_px"]) > 0
    edges = (prob >= p["edge_thr"]) & obj_band
    skel = _skeletonize(edges)
    n, lbl = cv2.connectedComponents(skel.astype(np.uint8))
    prims = []
    for i in range(1, n):
        ys, xs = np.where(lbl == i)
        if len(xs) < p["min_len"]:
            continue
        prim = _fit_fragment(xs, ys, p["straight_thr"])
        ec = float(prob[ys, xs].mean()) * 100.0
        if ec < p["edge_conf_min_raw"]:
            continue
        m = np.zeros((H, W), np.uint8); m[ys, xs] = 255
        prim.update(edge_confidence=ec, mask=m,
                    centroid=(float(xs.mean()), float(ys.mean())))
        prims.append(prim)
    prims.sort(key=lambda a: -(a["edge_confidence"] * a["length"]))
    return prims[:p["max_prims"]]


def classify_location(prim_mask, perim_band, interior, on_perim_min, inside_min):
    """Is a fragment ON the body perimeter (outer) or INSIDE the body (interior)?
    This is what makes a boundary a boundary -- location, not edge strength."""
    ys, xs = np.where(prim_mask > 0)
    if len(xs) == 0:
        return "interior", 0.0
    on_perim = float(perim_band[ys, xs].mean())
    inside = float(interior[ys, xs].mean())
    if on_perim >= on_perim_min:
        return "outer", on_perim
    if inside >= inside_min:
        return "interior", on_perim
    return ("outer" if on_perim >= inside else "interior"), on_perim


def semantic_support_along(prim_mask, support_up):
    """Mean object-support (0-100) over a primitive's pixels."""
    sel = prim_mask > 0
    return float(support_up[sel].mean()) if sel.any() else 0.0


# ============================================================ STAGE 3: boundary primitives
def ellipse_boundary(body, support_up, prob, crop_hw, thickness=3):
    """The body ELLIPSE perimeter = the PRIMARY boundary primitive. Robust: a
    clean oval that won't chase mask leaks into the seabed. edge_confidence =
    PiDiNet along the ellipse rim; semantic_support = support along it."""
    ch, cw = crop_hw
    band = _ellipse_perim_band((ch, cw), body["ellipse"], 1.0, thickness=thickness)
    if not band.any():
        return None
    m = (band.astype(np.uint8)) * 255
    ys, xs = np.where(m > 0)
    return dict(source="ellipse", ptype="ellipse_perimeter", from_ref="g1",
                mask=m, bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                length=None, edge_confidence=float(prob[band].mean()) * 100.0,
                semantic_support=float(support_up[band].mean()),
                centroid=(float(xs.mean()), float(ys.mean())),
                orient=None, endpoints=None)


def silhouette_boundary(mask_u8, support_up, prob, silhouette_eps):
    """The body SILHOUETTE = a SECONDARY boundary primitive. More detailed than
    the ellipse, but trusts every mask wobble including outward leaks; kept so
    Stage 4 can prefer it where the body is genuinely non-elliptical."""
    H, W = mask_u8.shape
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    eps = silhouette_eps * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    m = np.zeros((H, W), np.uint8)
    cv2.drawContours(m, [approx], -1, 255, 2)
    ys, xs = np.where(m > 0)
    return dict(source="silhouette", ptype="silhouette", from_ref="r1",
                mask=m, bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                length=int(cv2.arcLength(approx, True)),
                edge_confidence=float(prob[ys, xs].mean()) * 100.0,
                semantic_support=float(support_up[ys, xs].mean()),
                centroid=(float(xs.mean()), float(ys.mean())),
                orient=None, endpoints=None)


def build_boundaries(ellipse_b, silh, geoms, p):
    """Boundary primitives, in priority order:
      b1 = ellipse perimeter (PRIMARY, robust)
      b2 = silhouette        (SECONDARY, detailed)
      b3.. = OUTER PiDiNet fragments that clear the edge gate (corroboration)
    Interior fragments are NOT boundaries. id_b is assigned here, in order."""
    boundaries = [prim for prim in (ellipse_b, silh) if prim is not None]
    outer = [g for g in geoms
             if g["location"] == "outer" and g["edge_confidence"] >= p["edge_conf_min"]]
    for g in outer:
        boundaries.append(dict(source="edge", from_ref=g["id"], ptype=g["ptype"],
                               mask=g["mask"], bbox=g["bbox"],
                               edge_confidence=g["edge_confidence"],
                               semantic_support=g["semantic_support"],
                               length=g.get("length"), orient=g.get("orient"),
                               endpoints=g.get("endpoints")))
    for i, b in enumerate(boundaries, 1):
        b["id_b"] = f"b{i}"
    return boundaries


# ============================================================ relations
def _adiff(a, b):
    if a is None or b is None:
        return None
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def boundary_relations(boundaries, body_id, p):
    """on_perimeter to the body for every boundary; parallel/collinear among
    boundary line primitives."""
    rel = [("on_perimeter", b["id_b"], body_id) for b in boundaries]
    segs = [b for b in boundaries if b["ptype"] == "line_segment" and b.get("endpoints")]
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            A, B = segs[i], segs[j]
            da = _adiff(A.get("orient"), B.get("orient"))
            if da is None:
                continue
            if da < p["parallel_deg"]:
                rel.append(("parallel", A["id_b"], B["id_b"]))
            (ax1, ay1), (ax2, ay2) = A["endpoints"]
            (bx1, by1), _ = B["endpoints"]
            dx, dy = ax2 - ax1, ay2 - ay1
            L = math.hypot(dx, dy) + 1e-6
            perp = abs((bx1 - ax1) * (-dy) + (by1 - ay1) * dx) / L
            if da < p["parallel_deg"] and perp < p["collinear_px"]:
                rel.append(("collinear", A["id_b"], B["id_b"]))
    return rel


# ============================================================ foreign objects
def foreign_object_regions(crop_rgb, ellipse, inner_mask, prob, tok, grid,
                           support_up, body_feat, body_area, p):
    """Bright foreign surfaces inside the body (m4.shell_atoms), relabelled as
    generic foreign_object features. merged_uncertain flags implausibly large /
    elongated regions (the close-objects-fuse-into-one case; fix deferred)."""
    shp = dict(shell_bright_k=p["fo_bright_k"], shell_min_frac=p["fo_min_frac"],
               shell_min_pidi=p["fo_min_pidi"], max_shells=p["fo_max"])
    raw = shell_atoms(crop_rgb, ellipse, inner_mask, prob, tok, grid,
                      support_up, body_feat, shp)
    out = []
    for k, s in enumerate(raw, 1):
        x1, y1, x2, y2 = s["bbox"]
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        aspect = max(w, h) / float(min(w, h))
        merged = (s["area"] > p["fo_merge_area_frac"] * body_area) or \
                 (aspect > p["fo_merge_aspect"])
        out.append(dict(id=f"f{k}", role="foreign_object",
                        centroid=s["centroid"], bbox=s["bbox"], mask=s["mask"],
                        area=int(s["area"]), contrast=float(s["contrast"]),
                        edge_strength=float(s["pidi"]),
                        semantic_support=float(s["support"]),
                        merged_uncertain=bool(merged)))
    return out


# ============================================================ facts
def write_facts(path, det_box_crop, body, foreigns, geoms, boundaries, relations,
                oid="o1", cls="animal_urchin"):
    L = ["% urchin decomposition (spec stages 1-2-3). SPINES SKIPPED.",
         "% boundary = silhouette (semantic) + corroborating OUTER PiDiNet fragments.",
         "% interior PiDiNet fragments kept as geometric_primitive location(_,interior).",
         ""]
    bx = [int(round(v)) for v in det_box_crop]
    L += [f"detection({oid}).", f"class({oid},{cls}).",
          f"proposal_box({oid},box({bx[0]},{bx[1]},{bx[2]},{bx[3]}))."]

    # ---------- STAGE 1: semantic regions ----------
    L += ["", "% --- Stage 1: semantic regions ---"]
    rcx, rcy = [int(round(v)) for v in body["centroid"]]
    rb = [int(round(v)) for v in body["bbox"]]
    barea = int((body["mask"] > 0).sum())
    L += [f"semantic_region(r1).", f"belongs_to_detection(r1,{oid}).",
          f"region_role(r1,body).",
          f"region_centroid(r1,point({rcx},{rcy})).",
          f"region_box(r1,box({rb[0]},{rb[1]},{rb[2]},{rb[3]})).",
          f"region_area(r1,{barea}).",
          f"semantic_support(r1,{int(round(body['support']))}).",
          f"region_objectness(r1,{int(round(body['objness']))}).",
          f"region_coherence(r1,{int(round(body['coher']))}).",
          f"inside(r1,{oid})."]
    for f in foreigns:
        fcx, fcy = [int(round(v)) for v in f["centroid"]]
        fb = [int(round(v)) for v in f["bbox"]]
        L += ["", f"semantic_region({f['id']}).",
              f"belongs_to_detection({f['id']},{oid}).",
              f"region_role({f['id']},foreign_object).",
              f"region_centroid({f['id']},point({fcx},{fcy})).",
              f"region_box({f['id']},box({fb[0]},{fb[1]},{fb[2]},{fb[3]})).",
              f"region_area({f['id']},{f['area']}).",
              f"semantic_support({f['id']},{int(round(f['semantic_support']))}).",
              f"feature_contrast({f['id']},{int(round(f['contrast']))}).",
              f"edge_strength({f['id']},{int(round(f['edge_strength']))}).",
              f"on_region({f['id']},r1).", f"inside({f['id']},{oid})."]
        if f["merged_uncertain"]:
            L.append(f"merged_uncertain({f['id']}).")

    # ---------- STAGE 2: geometric primitives ----------
    L += ["", "% --- Stage 2: geometric primitives ---"]
    ecx, ecy, ed1, ed2, eang = body["ellipse"]
    eb = [int(round(v)) for v in body["bbox"]]
    L += [f"geometric_primitive(g1).", f"belongs_to_detection(g1,{oid}).",
          f"primitive_type(g1,ellipse).",
          f"ellipse(g1,point({int(round(ecx))},{int(round(ecy))}),"
          f"axes({int(round(ed1))},{int(round(ed2))}),angle({int(round(eang))})).",
          f"primitive_box(g1,box({eb[0]},{eb[1]},{eb[2]},{eb[3]})).",
          f"fit_error(g1,{int(round(body['fit_error']*100))}).",
          f"edge_confidence(g1,{int(round(body.get('ellipse_edge_conf',0)))}).",
          f"describes_region(g1,r1)."]
    for g in geoms:
        gb = [int(round(v)) for v in g["bbox"]]
        L += ["", f"geometric_primitive({g['id']}).",
              f"belongs_to_detection({g['id']},{oid}).",
              f"primitive_type({g['id']},{g['ptype']}).",
              f"primitive_box({g['id']},box({gb[0]},{gb[1]},{gb[2]},{gb[3]})).",
              f"length({g['id']},{int(round(g['length']))}).",
              f"orientation({g['id']},{int(round(g['orient']))}).",
              f"fit_error({g['id']},{int(round(g['fit_error']*100))}).",
              f"edge_confidence({g['id']},{int(round(g['edge_confidence']))}).",
              f"semantic_support({g['id']},{int(round(g['semantic_support']))}).",
              f"location({g['id']},{g['location']})."]
        if g.get("endpoints"):
            (x1, y1), (x2, y2) = g["endpoints"]
            L.append(f"endpoints({g['id']},point({x1},{y1}),point({x2},{y2})).")

    # ---------- STAGE 3: boundary primitives ----------
    L += ["", "% --- Stage 3: boundary primitives (silhouette + outer corroboration) ---"]
    for b in boundaries:
        bb = [int(round(v)) for v in b["bbox"]]
        L += ["", f"boundary_primitive({b['id_b']}).",
              f"belongs_to_detection({b['id_b']},{oid}).",
              f"boundary_source({b['id_b']},{b['source']}).",
              f"from_ref({b['id_b']},{b['from_ref']}).",
              f"primitive_type({b['id_b']},{b['ptype']}).",
              f"primitive_box({b['id_b']},box({bb[0]},{bb[1]},{bb[2]},{bb[3]})).",
              f"edge_confidence({b['id_b']},{int(round(b['edge_confidence']))}).",
              f"semantic_support({b['id_b']},{int(round(b['semantic_support']))}).",
              f"inside({b['id_b']},{oid})."]
        if b.get("length") is not None:
            L.append(f"length({b['id_b']},{int(round(b['length']))}).")

    # ---------- relations ----------
    L += ["", "% --- relations ---"]
    for r, x, y in relations:
        L.append(f"{r}({x},{y}).")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ============================================================ visualize
def visualize(crop_rgb, det_box_crop, body, foreigns, geoms, boundaries, out_path, title):
    H, W = crop_rgb.shape[:2]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    bx1, by1, bx2, by2 = det_box_crop

    ax[0].imshow(crop_rgb)
    ax[0].add_patch(Rectangle((bx1, by1), bx2-bx1, by2-by1, fill=False, ec="lime", lw=2))
    ov = np.zeros((H, W, 4)); ov[body["mask"] > 0] = [1, 1, 1, 0.22]; ax[0].imshow(ov)
    ax[0].set_title("crop + DETR box + support mask")

    ax[1].imshow(crop_rgb)
    ax[1].contour(body["mask"].astype(float), [127], colors="yellow", linewidths=1.6)
    for f in foreigns:
        c = "red" if f["merged_uncertain"] else "magenta"
        ax[1].contour(f["mask"].astype(float), [127], colors=c, linewidths=1.4)
    ax[1].set_title("Stage 1: body (yellow) + foreign objs (magenta; red=merged?)")

    ax[2].imshow(crop_rgb)
    em = ellipse_to_mask((H, W), body["ellipse"], 1.0)
    ax[2].contour(em.astype(float), [127], colors="lime", linewidths=1.0)
    for g in geoms:
        ys, xs = np.where(g["mask"] > 0)
        c = "cyan" if g["location"] == "outer" else "orange"   # outer vs interior
        ax[2].scatter(xs, ys, s=3, c=c, marker="s")
    ax[2].set_title("Stage 2: ellipse (lime) | outer (cyan) / interior (orange)")

    ax[3].imshow(crop_rgb)
    _bcol = {"ellipse": "lime", "silhouette": "orange", "edge": "yellow"}
    for b in boundaries:
        ys, xs = np.where(b["mask"] > 0)
        ax[3].scatter(xs, ys, s=3, c=_bcol.get(b["source"], "yellow"), marker="s")
    ax[3].set_title(f"Stage 3: ellipse b1 (lime) + silhouette b2 (orange) + edge (yellow) [{len(boundaries)}]")

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
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--min-grid", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--conn", type=int, choices=[4, 8], default=8)
    # PiDiNet
    ap.add_argument("--pidinet-weights", default="/srv/data1/Salim/Underwater/table7_pidinet.pth")
    ap.add_argument("--pidinet-device", default=None)
    ap.add_argument("--edge-thr", type=float, default=0.4)
    # geometric primitives
    ap.add_argument("--obj-band-px", type=int, default=5)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--straight-thr", type=float, default=0.08)
    ap.add_argument("--edge-conf-min-raw", type=float, default=20.0)
    ap.add_argument("--max-prims", type=int, default=40)
    # location classification (outer vs interior)
    ap.add_argument("--perim-px", type=int, default=3)
    ap.add_argument("--on-perim-min", type=float, default=0.5)
    ap.add_argument("--inside-min", type=float, default=0.6)
    ap.add_argument("--silhouette-eps", type=float, default=0.01)
    # boundary gate (Stage 3): outer fragments need this much edge_confidence
    ap.add_argument("--edge-conf-min", type=float, default=35.0)
    # relations
    ap.add_argument("--parallel-deg", type=float, default=12.0)
    ap.add_argument("--collinear-px", type=float, default=4.0)
    # foreign objects
    ap.add_argument("--annulus-scale", type=float, default=1.8)
    ap.add_argument("--fo-bright-k", type=float, default=1.0)
    ap.add_argument("--fo-min-frac", type=float, default=0.03)
    ap.add_argument("--fo-min-pidi", type=float, default=0.06)
    ap.add_argument("--fo-max", type=int, default=4)
    ap.add_argument("--fo-merge-area-frac", type=float, default=0.30)
    ap.add_argument("--fo-merge-aspect", type=float, default=3.5)
    ap.add_argument("--no-foreign", action="store_true")
    # io
    ap.add_argument("--max-dets", type=int, default=40)
    ap.add_argument("--out-dir", default="./urchin_lp")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from pidinet_edge import PiDiNetEdger
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.fresh and os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    id2file, name2cat, cat2name, _ = core.load_gt_index(args.gt_json)
    if args.class_name not in name2cat:
        raise SystemExit(f"'{args.class_name}' not in {list(name2cat)[:12]}...")
    target = name2cat[args.class_name]
    print(f"Target '{args.class_name}' -> id={target}")

    model = core.load_dinov3(args.repo_root, args.weights, args.weights_kind,
                             prefer_ema=not args.no_ema, device=device)
    edger = PiDiNetEdger(args.pidinet_weights, device=args.pidinet_device or device)
    by_img = core.load_predictions(args.pred_json, args.box_format)

    gp = dict(edge_thr=args.edge_thr, obj_band_px=args.obj_band_px, min_len=args.min_len,
              straight_thr=args.straight_thr, edge_conf_min_raw=args.edge_conf_min_raw,
              max_prims=args.max_prims)
    bp = dict(edge_conf_min=args.edge_conf_min)
    rp = dict(parallel_deg=args.parallel_deg, collinear_px=args.collinear_px)
    fp = dict(fo_bright_k=args.fo_bright_k, fo_min_frac=args.fo_min_frac,
              fo_min_pidi=args.fo_min_pidi, fo_max=args.fo_max,
              fo_merge_area_frac=args.fo_merge_area_frac, fo_merge_aspect=args.fo_merge_aspect)

    n_done, n_bound, n_outer, n_interior, n_fo, n_merged = 0, 0, 0, 0, 0, 0
    for image_id, dets in by_img.items():
        dets = [d for d in dets if d["cat"] == target and d["score"] >= args.score_thr]
        if not dets:
            continue
        fname = id2file.get(image_id)
        ipath = os.path.join(args.img_root, fname) if fname else None
        if not ipath or not os.path.exists(ipath):
            continue
        image_rgb = np.asarray(Image.open(ipath).convert("RGB"))
        for k, d in enumerate(dets):
            crop_rgb, (ox1, oy1, _, _) = core.crop_with_margin(image_rgb, d["box"], args.margin)
            if crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 6:
                continue
            ch, cw = crop_rgb.shape[:2]
            stem = f"{os.path.splitext(os.path.basename(fname))[0]}_det{k}"
            dbx = (d["box"][0]-ox1, d["box"][1]-oy1, d["box"][2]-ox1, d["box"][3]-oy1)
            ten, mhw = core.to_model_input(crop_rgb, args.min_grid, args.max_side, device)
            tok, grid = core.extract_tokens(model, ten, args.layer)
            support, gate = core.compute_support(tok, grid, mhw, dbx, (ch, cw), args.seed_frac)
            mask_grid = core.clean_mask(support, args.sup_thr_frac)
            if mask_grid is None:
                continue
            support_up = core.upsample_grid(support, (ch, cw), "bilinear")
            obj_proto, bg_proto = core.object_bg_prototypes(
                tok, grid, mhw, dbx, (ch, cw), args.seed_frac)
            mask_u8 = (core.upsample_mask(mask_grid, (ch, cw)).astype(np.uint8)) * 255

            body = ellipse_body_atom(mask_u8, support, None, (ch, cw), tok, grid,
                                     mask_grid, obj_proto, bg_proto, ridge_snap=False)
            if body is None:
                continue

            prob = edger.prob(crop_rgb)

            # ---- Stage 2: geometric primitives, classified outer/interior ----
            geoms = geometric_primitives(prob, mask_u8, gp)
            ker = np.ones((3, 3), np.uint8)
            mbin = (mask_u8 > 0).astype(np.uint8)
            interior_band = cv2.erode(mbin, ker, iterations=args.perim_px) > 0
            perim_band = (cv2.dilate(mbin, ker, iterations=args.perim_px) -
                          cv2.erode(mbin, ker, iterations=args.perim_px)) > 0
            for gi, g in enumerate(geoms, 2):           # g1 is the ellipse
                g["id"] = f"g{gi}"
                loc, _ = classify_location(g["mask"], perim_band, interior_band,
                                           args.on_perim_min, args.inside_min)
                g["location"] = loc
                g["semantic_support"] = semantic_support_along(g["mask"], support_up)

            # ---- Stage 3: boundary primitives (ellipse primary, silhouette secondary) ----
            ellipse_b = ellipse_boundary(body, support_up, prob, (ch, cw),
                                         thickness=max(2, args.perim_px))
            silh = silhouette_boundary(mask_u8, support_up, prob, args.silhouette_eps)
            boundaries = build_boundaries(ellipse_b, silh, geoms, bp)
            body["ellipse_edge_conf"] = (ellipse_b["edge_confidence"]
                                         if ellipse_b is not None else 0.0)

            # ---- foreign objects (Stage 1 features) ----
            foreigns = []
            if not args.no_foreign:
                inner, _ = annulus_zone(body["ellipse"], (ch, cw), dbx, args.annulus_scale)
                foreigns = foreign_object_regions(
                    crop_rgb, body["ellipse"], inner, prob, tok, grid, support_up,
                    body.get("rfeat"), float((mask_u8 > 0).sum()), fp)

            relations = boundary_relations(boundaries, "r1", rp)
            relations += [("on_region", f["id"], "r1") for f in foreigns]

            write_facts(os.path.join(args.out_dir, stem + ".lp"), dbx, body, foreigns,
                        geoms, boundaries, relations, cls=args.class_name)
            visualize(crop_rgb, dbx, body, foreigns, geoms, boundaries,
                      os.path.join(args.out_dir, stem + ".png"),
                      title=f"{fname} det#{k} s={d['score']:.2f} | regions=1+{len(foreigns)} "
                            f"prims={len(geoms)+1} boundaries={len(boundaries)} "
                            f"fit_err={body['fit_error']:.2f} obj={body['objness']:.0f}")
            n_done += 1; n_bound += len(boundaries); n_fo += len(foreigns)
            n_outer += sum(g["location"] == "outer" for g in geoms)
            n_interior += sum(g["location"] == "interior" for g in geoms)
            n_merged += sum(f["merged_uncertain"] for f in foreigns)
            if n_done >= args.max_dets:
                break
        if n_done >= args.max_dets:
            break

    print(f"\nWrote {n_done} urchin decompositions to {args.out_dir}")
    if n_done:
        print(f"  boundary primitives / det : {n_bound / n_done:.1f}  (b1=ellipse, b2=silhouette)")
        print(f"  PiDiNet fragments / det   : outer {n_outer/n_done:.1f}  interior {n_interior/n_done:.1f}")
        print(f"  foreign objects   / det   : {n_fo / n_done:.2f}  (merged_uncertain: {n_merged})")
        print("  boundary: b1 ellipse (primary) + b2 silhouette (secondary) + outer corroboration.")
        print("  spines: SKIPPED (deferred). PiDiNet role: boundary corroboration + foreign objects.")


if __name__ == "__main__":
    main()