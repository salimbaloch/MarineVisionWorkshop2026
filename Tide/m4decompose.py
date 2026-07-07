#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m4decompose.py  —  UNIFIED Stage 1-2-3 visual decomposition, one driver.

Collapses decompose.py (urchin) + fish_probe.py (fish) into a single crop->.lp
pipeline selected by --profile {urchin,fish,bottle}. Every profile emits the SAME
atom vocabulary, so one Stage-4 ASP program reads all three:

    detection / class / proposal_box
    semantic_region(role=body | foreign_object)   + support/objectness/coherence
    geometric_primitive(type=ellipse|rect|line_segment|curve_segment)
    boundary_primitive(source=silhouette|ellipse|edge)  + edge_conf/semantic_support
    relations

Everything class-specific is a VALUE, not a new predicate. The knobs live in a
Profile; the machinery (support, B_dino, PiDiNet, ellipse fit) is shared.

PROFILES
--------
urchin  body = ellipse (radial blob). Boundary ellipse-primary + silhouette +
        outer PiDiNet corroboration. Foreign objects ON (shell rims). Spines
        SKIPPED (PiDiNet finds no gradient on turbid spines).
fish    body = support MASK (an ellipse would crop the fins). Boundary
        silhouette-primary. Foreign OFF. Emits routing DIAGNOSTICS
        (bimodality/solidity/elongation/concavities) only -- parts are NOT
        forced into atoms; Stage-4 may split r1 iff the evidence supports it.
        (The head/tail/fin partition of fish_parts.py is deliberately dropped:
        it imposed a template with no evidence.)
bottle  body = ellipse, with B_dino RIDGE-SNAP ON (the support mask under-
        segments bright rigid objects). Boundary PiDiNet-PRIMARY (man-made
        objects have a real edge; where they don't -- turbid/encrusted -- the
        silhouette is kept as fallback, so the boundary source is effectively
        evidence-routed within the class). Adds a rotated-RECT region primitive.
        Foreign OFF.

Data flow (answers "does it re-crop?"): identical for all three. Point it at one
predictions dump; --class-name selects the category; crops are made on the fly
via core.crop_with_margin from (image, box). No pre-saved crops, no separate
bottle-extraction step -- bottle crops are produced exactly like urchin ones.

Run on your machine. Deps: torch, numpy, cv2, PIL, scipy, matplotlib,
(skimage for skeletonize, optional), core.py, m2_regions.py, m4_decompose.py,
pidinet_edge.py + pidinet_pkg/ + table7_pidinet.pth.
"""
import os
import math
import shutil
import argparse
from dataclasses import dataclass, field
from typing import Tuple

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


# ============================================================ profiles
@dataclass
class Profile:
    cls_default:    str
    body_geom:      str            # "ellipse" | "mask"
    boundary_order: Tuple[str, ...]  # order over {"ellipse","silhouette","frags"} -> b1,b2,...
    ridge_snap:     bool
    foreign:        bool
    parts_probe:    bool
    edge_gate:      str = "mask"    # "mask" | "mask_or_ridge"  how an edge frag earns "outer"
    region_prims:   Tuple[str, ...] = field(default_factory=tuple)  # extra region descriptors, e.g. ("rect",)


# edge_gate: "mask"          -> outer iff on the mask perimeter band (original rule).
#            "mask_or_ridge" -> ALSO outer if the fragment rides the B_dino ridge, so a
#                               true outline edge sitting INSIDE an over-shooting support
#                               mask still counts as a boundary. urchin stays "mask"
#                               because there B_dino marks the spine halo, not the body edge.
PROFILES = {
    "urchin": Profile("animal_urchin", "ellipse",
                      ("ellipse", "silhouette", "frags"),
                      ridge_snap=False, foreign=True,  parts_probe=False,
                      edge_gate="mask"),
    "fish":   Profile("animal_fish",   "mask",
                      ("silhouette", "ellipse", "frags"),
                      ridge_snap=False, foreign=False, parts_probe=True,
                      edge_gate="mask_or_ridge"),
    "bottle": Profile("bottle_plastic", "ellipse",
                      ("frags", "silhouette", "ellipse"),
                      ridge_snap=True,  foreign=False, parts_probe=False,
                      edge_gate="mask_or_ridge", region_prims=("rect",)),
}


# ============================================================ skeletonize
def _skeletonize(binmask):
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
    """PCA line/curve fit to a connected edge fragment."""
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
    """PiDiNet prob -> threshold -> skeletonize -> CC -> fit line/curve, restricted
    to an object band. edge_confidence = mean PiDiNet prob along the fragment."""
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


def rect_primitive(mask_u8, support_up, prob):
    """Rotated bounding rectangle of the body mask -> a man-made region descriptor.
    fit_error = 1 - (area / rect_area) (0 == perfectly rectangular)."""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    if w < 1 or h < 1:
        return None
    area = float(cv2.contourArea(c))
    fit_err = 1.0 - area / (w * h + 1e-6)
    box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)
    m = np.zeros_like(mask_u8)
    cv2.drawContours(m, [box], -1, 255, 2)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return dict(ptype="rect", center=(cx, cy), size=(w, h), angle=ang % 180.0,
                bbox=bbox, length=float(max(w, h)), orient=float(ang % 180.0),
                fit_error=float(max(0.0, min(1.0, fit_err))),
                edge_confidence=float(prob[ys, xs].mean()) * 100.0,
                semantic_support=float(support_up[ys, xs].mean()), mask=m)


def classify_location(prim_mask, perim_band, interior, on_perim_min, inside_min,
                      bdino_n=None, ridge_thr=None):
    """Is a fragment ON the body perimeter (outer) or INSIDE the body (interior)?
    Two routes to 'outer':
      1. mask route  -- the fragment sits on the mask perimeter band.
      2. ridge route -- (only when bdino_n/ridge_thr given) the fragment rides the
         independent B_dino boundary ridge. This rescues a true outline edge that
         falls INSIDE an over-shooting support mask (the bottle g3 case), which the
         mask route alone buries as interior."""
    ys, xs = np.where(prim_mask > 0)
    if len(xs) == 0:
        return "interior", 0.0
    on_perim = float(perim_band[ys, xs].mean())
    inside = float(interior[ys, xs].mean())
    if bdino_n is not None and ridge_thr is not None:
        ridge = float(bdino_n[ys, xs].mean())          # normalized 0..1 along the fragment
        if ridge >= ridge_thr:
            return "outer", max(on_perim, ridge)
    if on_perim >= on_perim_min:
        return "outer", on_perim
    if inside >= inside_min:
        return "interior", on_perim
    return ("outer" if on_perim >= inside else "interior"), on_perim


def semantic_support_along(prim_mask, support_up):
    sel = prim_mask > 0
    return float(support_up[sel].mean()) if sel.any() else 0.0


# ============================================================ STAGE 3: boundary primitives
def ellipse_boundary(body, support_up, prob, crop_hw, thickness=3):
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


def assemble_boundaries(ellipse_b, silh, geoms, order, p):
    """Boundary primitives assembled in the profile's priority order. Whichever
    source leads becomes b1. OUTER PiDiNet fragments that clear the edge gate are
    the 'frags' source; interior fragments are never boundaries."""
    frags = []
    for g in geoms:
        if g.get("location") == "outer" and g["edge_confidence"] >= p["edge_conf_min"]:
            frags.append(dict(source="edge", from_ref=g["id"], ptype=g["ptype"],
                              mask=g["mask"], bbox=g["bbox"],
                              edge_confidence=g["edge_confidence"],
                              semantic_support=g["semantic_support"],
                              length=g.get("length"), orient=g.get("orient"),
                              endpoints=g.get("endpoints")))
    src = {"ellipse":    [ellipse_b] if ellipse_b is not None else [],
           "silhouette": [silh] if silh is not None else [],
           "frags":      frags}
    boundaries = []
    for key in order:
        boundaries.extend(src.get(key, []))
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


# ============================================================ facts (unified)
def _clamp(v):
    return int(max(0, min(100, round(v))))

def _conf_body(r):        return _clamp(0.4*r["objness"] + 0.4*r["support"] + 0.2*r["coher"])
def _conf_foreign(f):     return _clamp(0.7*f["semantic_support"] + 0.3*min(100.0, f["contrast"]+f["edge_strength"]))
def _conf_shape(fit_err): return _clamp((1.0 - fit_err) * 100.0)          # ellipse / rect: fit quality
def _conf_frag(g):        return _clamp(0.6*g["edge_confidence"] + 0.4*g["semantic_support"])
def _conf_boundary(b):    return _clamp(0.5*b["edge_confidence"] + 0.5*b["semantic_support"])



def _poly_from_mask(mask, max_pts=40, eps=1.5):
    """Largest-contour polyline of an atom mask, simplified and capped -- emitted
    as a % poly comment so render_channels can stroke the true geometry.
    Comments are invisible to clingo: zero grounding cost."""
    if mask is None:
        return None
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 4:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea if len(cnts) > 1 else len)
    c = cv2.approxPolyDP(c, eps, closed=False).reshape(-1, 2)
    if len(c) > max_pts:
        c = c[np.linspace(0, len(c) - 1, max_pts).astype(int)]
    return [(int(x), int(y)) for x, y in c]


def write_facts(path, det_box_crop, crop_hw, region, foreigns, region_prims, geoms,
                boundaries, relations, diag, oid="o1", cls="object"):
    bx = [int(round(v)) for v in det_box_crop]
    ch, cw = crop_hw
    L = [f"detection({oid}).", f"class({oid},{cls}).",
         f"proposal_box({oid},box({bx[0]},{bx[1]},{bx[2]},{bx[3]})).",
         f"crop_size({oid},size({int(ch)},{int(cw)}))."]

    # detection-level routing diagnostics (fish); no comment header
    if diag is not None:
        L += [f"bimodality({oid},{int(round(diag['bimodality']*100))}).",
              f"feature_dispersion({oid},{int(round(diag['dispersion']*100))}).",
              f"region_solidity({oid},{int(round(diag['solidity']*100))}).",
              f"region_elongation({oid},{int(round(diag['elongation']*100))}).",
              f"concavities({oid},{diag['n_defects']}).",
              f"ellipse_fit_error({oid},{int(round(diag['fit_error']*100))})."]
        if diag.get("fins_held"):
            L.append(f"fins_held_candidate({oid}).")
        if diag.get("bimodality", 0.0) >= diag.get("bimod_thr", 1e9):
            L.append(f"multi_material_candidate({oid}).")

    # ---------- Stage 1 ----------
    L += ["", "% Stage 1: Semantic Regions"]
    rcx, rcy = [int(round(v)) for v in region["centroid"]]
    rb = [int(round(v)) for v in region["bbox"]]
    L += [f"semantic_region(r1).", f"belongs_to_detection(r1,{oid}).",
          f"region_role(r1,body).",
          f"region_centroid(r1,point({rcx},{rcy})).",
          f"region_box(r1,box({rb[0]},{rb[1]},{rb[2]},{rb[3]})).",
          f"region_area(r1,{int(region['area'])}).",
          f"semantic_support(r1,{int(round(region['support']))}).",
          f"region_objectness(r1,{int(round(region['objness']))}).",
          f"region_coherence(r1,{int(round(region['coher']))}).",
          f"confidence(r1,{_conf_body(region)}).",
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
              f"confidence({f['id']},{_conf_foreign(f)}).",
              f"on_region({f['id']},r1).", f"inside({f['id']},{oid})."]
        if f["merged_uncertain"]:
            L.append(f"merged_uncertain({f['id']}).")

    # ---------- Stage 2 ----------
    L += ["", "% Stage 2: Geometric Primitives"]
    for rp in region_prims:                 # region-describing primitives (g1 ellipse, g2 rect, ...)
        gid = rp["id"]
        gb = [int(round(v)) for v in rp["bbox"]]
        L += ["", f"geometric_primitive({gid}).",
              f"belongs_to_detection({gid},{oid}).",
              f"primitive_type({gid},{rp['ptype']}).",
              f"primitive_box({gid},box({gb[0]},{gb[1]},{gb[2]},{gb[3]})).",
              f"fit_error({gid},{int(round(rp['fit_error']*100))}).",
              f"edge_confidence({gid},{int(round(rp.get('edge_confidence',0)))}).",
              f"confidence({gid},{_conf_shape(rp['fit_error'])}).",
              f"describes_region({gid},r1)."]
        if rp["ptype"] == "ellipse":
            ecx, ecy, ed1, ed2, eang = rp["ellipse"]
            L.append(f"ellipse({gid},point({int(round(ecx))},{int(round(ecy))}),"
                     f"axes({int(round(ed1))},{int(round(ed2))}),angle({int(round(eang))})).")
        elif rp["ptype"] == "rect":
            cx, cy = rp["center"]; w, h = rp["size"]
            L.append(f"rect({gid},point({int(round(cx))},{int(round(cy))}),"
                     f"size({int(round(w))},{int(round(h))}),angle({int(round(rp['angle']))})).")
            L.append(f"semantic_support({gid},{int(round(rp.get('semantic_support',0)))}).")

    for g in geoms:                          # PiDiNet edge fragments
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
              f"confidence({g['id']},{_conf_frag(g)}).",
              f"location({g['id']},{g['location']})."]
        if g.get("endpoints"):
            (x1, y1), (x2, y2) = g["endpoints"]
            L.append(f"endpoints({g['id']},point({x1},{y1}),point({x2},{y2})).")

    # ---------- Stage 3 ----------
    L += ["", "% Stage 3: Boundary Primitives"]
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
              f"confidence({b['id_b']},{_conf_boundary(b)}).",
              f"inside({b['id_b']},{oid})."]
        if b.get("length") is not None:
            L.append(f"length({b['id_b']},{int(round(b['length']))}).")
        pts = _poly_from_mask(b.get("mask"))
        if pts:
            L.append("% poly " + b['id_b'] + ": " + " ".join(f"{x},{y}" for x, y in pts))

    # ---------- Relations ----------
    L += ["", "% Relations"]
    for r, x, y in relations:
        L.append(f"{r}({x},{y}).")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ============================================================ visualize (generic)
def visualize(crop_rgb, det_box_crop, region, support_up, foreigns, region_prims,
              geoms, boundaries, out_path, title):
    H, W = crop_rgb.shape[:2]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    bx1, by1, bx2, by2 = det_box_crop

    # panel 0: crop + DETR box + DINO support HEATMAP (+ mask outline)
    ax[0].imshow(crop_rgb)
    ax[0].imshow(support_up, cmap="jet", alpha=0.45, vmin=0, vmax=100)
    ax[0].add_patch(Rectangle((bx1, by1), bx2-bx1, by2-by1, fill=False, ec="lime", lw=2))
    ax[0].contour(region["mask"].astype(float), [127], colors="white", linewidths=1.0)
    ax[0].set_title("crop + DETR box + support heatmap")

    ax[1].imshow(crop_rgb)
    ax[1].contour(region["mask"].astype(float), [127], colors="yellow", linewidths=1.6)
    for f in foreigns:
        c = "red" if f["merged_uncertain"] else "magenta"
        ax[1].contour(f["mask"].astype(float), [127], colors=c, linewidths=1.4)
    ax[1].set_title(f"Stage 1: body (yellow) + foreign ({len(foreigns)})")

    # panel 2: region prims drawn from their PARAMETRIC form (filled -> contoured once,
    # so a rect/ellipse is a single outline, not a double-stroked one) + PiDiNet frags
    ax[2].imshow(crop_rgb)
    for rp in region_prims:
        m = np.zeros((H, W), np.uint8)
        if rp["ptype"] == "ellipse":
            cx, cy, d1, d2, ang = rp["ellipse"]
            cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                        (max(1, int(round(d1/2))), max(1, int(round(d2/2)))),
                        float(ang), 0, 360, 255, -1)
        elif rp["ptype"] == "rect":
            cx, cy = rp["center"]; w, h = rp["size"]
            box = cv2.boxPoints(((cx, cy), (w, h), rp["angle"])).astype(np.int32)
            cv2.fillPoly(m, [box], 255)
        else:
            continue
        col = "lime" if rp["ptype"] == "ellipse" else "deepskyblue"
        ax[2].contour(m.astype(float), [127], colors=col, linewidths=1.2)
    for g in geoms:
        ys, xs = np.where(g["mask"] > 0)
        c = "cyan" if g["location"] == "outer" else "orange"
        ax[2].scatter(xs, ys, s=3, c=c, marker="s")
    ax[2].set_title("Stage 2: ellipse (lime) / rect (blue) | outer (cyan) / interior (orange)")

    ax[3].imshow(crop_rgb)
    _bcol = {"ellipse": "lime", "silhouette": "orange", "edge": "yellow"}
    for b in boundaries:
        ys, xs = np.where(b["mask"] > 0)
        ax[3].scatter(xs, ys, s=3, c=_bcol.get(b["source"], "yellow"), marker="s")
    ax[3].set_title(f"Stage 3: boundaries in priority order [{len(boundaries)}]")

    for a in ax:
        a.set_xlim(0, W); a.set_ylim(H, 0); a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================================================ per-detection
def decompose_one(crop_rgb, dbx, tok, grid, mhw, prof, edger, args, device):
    """Run the shared substrate + profile-gated assembly on one crop. Returns a
    dict of everything write_facts/visualize need, or None to skip."""
    ch, cw = crop_rgb.shape[:2]
    support, gate = core.compute_support(tok, grid, mhw, dbx, (ch, cw), args.seed_frac)
    mask_grid = core.clean_mask(support, args.sup_thr_frac)
    if mask_grid is None:
        return None
    support_up = core.upsample_grid(support, (ch, cw), "bilinear")
    obj_proto, bg_proto = core.object_bg_prototypes(tok, grid, mhw, dbx, (ch, cw), args.seed_frac)
    mask_u8 = (core.upsample_mask(mask_grid, (ch, cw)).astype(np.uint8)) * 255

    # B_dino: needed for ridge-snap (bottle) and/or the ridge edge-gate (fish, bottle).
    # ellipse_body_atom no-ops snap if bdino_up is None.
    gate = args.edge_gate if getattr(args, "edge_gate", "profile") != "profile" else prof.edge_gate
    need_ridge = gate == "mask_or_ridge"
    bdino_up = None
    if prof.ridge_snap or need_ridge:
        bdino = core.feature_boundary_map(tok, grid, args.conn)
        bdino_up = core.upsample_grid(bdino, (ch, cw), "bilinear")

    body = ellipse_body_atom(mask_u8, support, bdino_up, (ch, cw), tok, grid,
                             mask_grid, obj_proto, bg_proto, ridge_snap=prof.ridge_snap)
    if body is None:
        return None

    prob = edger.prob(crop_rgb)

    # ---- region (Stage 1 body): ellipse-body vs mask-body ----
    desc = shape_descriptors(mask_u8)
    if prof.body_geom == "mask":
        if desc is None:
            return None
        region = dict(centroid=desc["centroid"], bbox=desc["bbox"],
                      area=int((mask_u8 > 0).sum()),
                      support=float(support_up[mask_u8 > 0].mean()),
                      objness=body["objness"], coher=body["coher"],
                      ellipse=body["ellipse"], ellipse_bbox=body["bbox"],
                      fit_error=body["fit_error"], mask=mask_u8)
    else:  # "ellipse"
        region = dict(centroid=body["centroid"], bbox=body["bbox"],
                      area=int((body["mask"] > 0).sum()),
                      support=float(body["support"]),
                      objness=body["objness"], coher=body["coher"],
                      ellipse=body["ellipse"], ellipse_bbox=body["bbox"],
                      fit_error=body["fit_error"], mask=body["mask"])

    # ---- routing diagnostics (fish) ----
    diag = None
    if prof.parts_probe and desc is not None:
        probe = core.feature_variance_probe(tok, grid, mask_grid)
        fins_held = (desc["solidity"] < args.solid_thr) or (desc["n_defects"] >= 2)
        diag = dict(bimodality=probe["bimodality"], dispersion=probe["dispersion"],
                    solidity=desc["solidity"], elongation=desc["elongation"],
                    n_defects=desc["n_defects"], fit_error=body["fit_error"],
                    fins_held=fins_held, bimod_thr=args.bimod_thr)

    # ---- Stage 2: PiDiNet fragments, classified outer/interior ----
    gp = dict(edge_thr=args.edge_thr, obj_band_px=args.obj_band_px, min_len=args.min_len,
              straight_thr=args.straight_thr, edge_conf_min_raw=args.edge_conf_min_raw,
              max_prims=args.max_prims)
    geoms = geometric_primitives(prob, mask_u8, gp)
    ker = np.ones((3, 3), np.uint8)
    mbin = (mask_u8 > 0).astype(np.uint8)
    interior_band = cv2.erode(mbin, ker, iterations=args.perim_px) > 0
    perim_band = (cv2.dilate(mbin, ker, iterations=args.perim_px) -
                  cv2.erode(mbin, ker, iterations=args.perim_px)) > 0

    # region-describing primitives (g1 ellipse always; g2 rect for bottle) -> fragments after
    ecx, ecy, ed1, ed2, eang = region["ellipse"]
    ellipse_perim_conf = float(prob[perim_band].mean()) * 100 if perim_band.any() else 0.0
    region_prims = [dict(id="g1", ptype="ellipse", ellipse=(ecx, ecy, ed1, ed2, eang),
                         bbox=region["ellipse_bbox"], fit_error=region["fit_error"],
                         edge_confidence=ellipse_perim_conf, mask=None)]
    if "rect" in prof.region_prims:
        rp = rect_primitive(mask_u8, support_up, prob)
        if rp is not None:
            rp["id"] = f"g{len(region_prims)+1}"
            region_prims.append(rp)

    # normalize B_dino to 0..1 within the object band, so the ridge gate threshold
    # is a fraction of the strongest boundary energy on THIS crop (scale-free).
    bdino_n, ridge_thr = None, None
    if need_ridge and bdino_up is not None:
        obj_band = cv2.dilate((mask_u8 > 0).astype(np.uint8),
                              np.ones((3, 3), np.uint8), iterations=args.obj_band_px) > 0
        peak = float(bdino_up[obj_band].max()) if obj_band.any() else float(bdino_up.max())
        bdino_n = np.clip(bdino_up / (peak + 1e-6), 0.0, 1.0)
        ridge_thr = args.ridge_gate_thr

    frag_start = len(region_prims) + 1
    for gi, g in enumerate(geoms, frag_start):
        g["id"] = f"g{gi}"
        loc, _ = classify_location(g["mask"], perim_band, interior_band,
                                   args.on_perim_min, args.inside_min,
                                   bdino_n=bdino_n, ridge_thr=ridge_thr)
        g["location"] = loc
        g["semantic_support"] = semantic_support_along(g["mask"], support_up)

    # ---- Stage 3: boundaries in profile priority order ----
    ellipse_b = ellipse_boundary(body, support_up, prob, (ch, cw), thickness=max(2, args.perim_px))
    silh = silhouette_boundary(mask_u8, support_up, prob, args.silhouette_eps)
    bp = dict(edge_conf_min=args.edge_conf_min)
    boundaries = assemble_boundaries(ellipse_b, silh, geoms, prof.boundary_order, bp)

    # ---- foreign objects (urchin) ----
    foreigns = []
    if prof.foreign:
        inner, _ = annulus_zone(body["ellipse"], (ch, cw), dbx, args.annulus_scale)
        fp = dict(fo_bright_k=args.fo_bright_k, fo_min_frac=args.fo_min_frac,
                  fo_min_pidi=args.fo_min_pidi, fo_max=args.fo_max,
                  fo_merge_area_frac=args.fo_merge_area_frac, fo_merge_aspect=args.fo_merge_aspect)
        foreigns = foreign_object_regions(crop_rgb, body["ellipse"], inner, prob, tok, grid,
                                          support_up, body.get("rfeat"),
                                          float((mask_u8 > 0).sum()), fp)

    # ---- relations ----
    rp = dict(parallel_deg=args.parallel_deg, collinear_px=args.collinear_px)
    relations = boundary_relations(boundaries, "r1", rp)
    relations += [("on_region", f["id"], "r1") for f in foreigns]

    return dict(region=region, foreigns=foreigns, region_prims=region_prims,
                geoms=geoms, boundaries=boundaries, relations=relations, diag=diag,
                body=body, support_up=support_up)


# ============================================================ GT matching (Stage-4/5 data)
def _iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt_boxes(gt_json):
    """{image_id: [(cat_id, [x1,y1,x2,y2]), ...]} from every bbox annotation."""
    import json
    from collections import defaultdict
    gt = json.load(open(gt_json))
    by = defaultdict(list)
    for a in gt.get("annotations", []):
        if "bbox" not in a:
            continue
        x, y, w, h = a["bbox"]
        by[a["image_id"]].append((a["category_id"], [x, y, x + w, y + h]))
    return by


def match_gt_crop(image_gt, target_cat, proposal_img, crop_off, crop_hw, min_iou):
    """Best GT box (same category, IoU>=min_iou with the image-coord proposal),
    returned in CROP coordinates. None if no match."""
    ox, oy = crop_off
    ch, cw = crop_hw
    best, best_iou = None, min_iou
    for cat, gbox in image_gt:
        if cat != target_cat:
            continue
        j = _iou_xyxy(proposal_img, gbox)
        if j >= best_iou:
            best, best_iou = gbox, j
    if best is None:
        return None
    x1, y1, x2, y2 = best
    cx1 = int(round(max(0, min(cw, x1 - ox))))
    cy1 = int(round(max(0, min(ch, y1 - oy))))
    cx2 = int(round(max(0, min(cw, x2 - ox))))
    cy2 = int(round(max(0, min(ch, y2 - oy))))
    return [cx1, cy1, cx2, cy2]


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=list(PROFILES))
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--weights-kind", choices=["auto", "pretrained", "deim"], default="deim")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--pred-json", "--pred", dest="pred_json", required=True)
    ap.add_argument("--gt-json", "--gt", dest="gt_json", required=True)
    ap.add_argument("--img-root", "--images", dest="img_root", required=True)
    ap.add_argument("--class-name", default=None,
                    help="dataset category to filter+emit. Defaults to the profile's class.")
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
    ap.add_argument("--edge-gate", choices=["profile", "mask", "mask_or_ridge"], default="profile",
                    help="override the profile's outer-gate (e.g. try ridge rescue on urchin)")
    # geometric primitives
    ap.add_argument("--obj-band-px", type=int, default=5)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--straight-thr", type=float, default=0.08)
    ap.add_argument("--edge-conf-min-raw", type=float, default=20.0)
    ap.add_argument("--max-prims", type=int, default=40)
    # location classification
    ap.add_argument("--perim-px", type=int, default=3)
    ap.add_argument("--on-perim-min", type=float, default=0.5)
    ap.add_argument("--inside-min", type=float, default=0.6)
    ap.add_argument("--ridge-gate-thr", type=float, default=0.45,
                    help="edge_gate=mask_or_ridge: frag is 'outer' if mean normalized "
                         "B_dino along it >= this (fraction of the crop's peak ridge).")
    ap.add_argument("--silhouette-eps", type=float, default=0.01)
    # boundary gate
    ap.add_argument("--edge-conf-min", type=float, default=35.0)
    # relations
    ap.add_argument("--parallel-deg", type=float, default=12.0)
    ap.add_argument("--collinear-px", type=float, default=4.0)
    # foreign objects (urchin)
    ap.add_argument("--annulus-scale", type=float, default=1.8)
    ap.add_argument("--fo-bright-k", type=float, default=1.0)
    ap.add_argument("--fo-min-frac", type=float, default=0.03)
    ap.add_argument("--fo-min-pidi", type=float, default=0.06)
    ap.add_argument("--fo-max", type=int, default=4)
    ap.add_argument("--fo-merge-area-frac", type=float, default=0.30)
    ap.add_argument("--fo-merge-aspect", type=float, default=3.5)
    # fish routing thresholds (only flip diagnostic flags; tune on your bench)
    ap.add_argument("--solid-thr", type=float, default=0.92)
    ap.add_argument("--bimod-thr", type=float, default=1.15)
    # io
    ap.add_argument("--max-dets", type=int, default=40)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--device", default=None)
    # Stage-4/5 data dumps
    ap.add_argument("--no-gt", action="store_true",
                    help="skip writing gt_boxes.json (GT-box matching for weight learning)")
    ap.add_argument("--gt-out", default=None, help="gt_boxes.json path (default <out_dir>/gt_boxes.json)")
    ap.add_argument("--gt-iou", type=float, default=0.3,
                    help="min IoU(proposal, GT) to accept a GT match, in image coords")
    ap.add_argument("--dump-emb", action="store_true",
                    help="dump per-detection DINOv3 token grid to <emb-dir>/<stem>.npz (key 'emb')")
    ap.add_argument("--emb-dir", default=None, help="embedding dir (default <out_dir>/emb)")
    args = ap.parse_args()

    prof = PROFILES[args.profile]
    if args.class_name is None:
        args.class_name = prof.cls_default
    if args.out_dir is None:
        args.out_dir = f"./{args.profile}_lp"

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
    print(f"[{args.profile}] target '{args.class_name}' -> id={target}  out={args.out_dir}")

    model = core.load_dinov3(args.repo_root, args.weights, args.weights_kind,
                             prefer_ema=not args.no_ema, device=device)
    edger = PiDiNetEdger(args.pidinet_weights, device=args.pidinet_device or device)
    by_img = core.load_predictions(args.pred_json, args.box_format)

    gt_by_img = {} if args.no_gt else load_gt_boxes(args.gt_json)
    gt_boxes = {}                                    # {stem: [x1,y1,x2,y2] in crop coords}
    crop_meta = {}                                   # {stem: [img_id, ox1, oy1, ix1,iy1,ix2,iy2, score]}
    if args.dump_emb:
        emb_dir = args.emb_dir or os.path.join(args.out_dir, "emb")
        os.makedirs(emb_dir, exist_ok=True)

    n_done = n_bound = n_outer = n_interior = n_fo = n_regprim = 0
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

            R = decompose_one(crop_rgb, dbx, tok, grid, mhw, prof, edger, args, device)
            if R is None:
                continue

            # --- Stage-4/5 data dumps ---
            crop_meta[stem] = [int(image_id), int(ox1), int(oy1),
                               float(d["box"][0]), float(d["box"][1]),
                               float(d["box"][2]), float(d["box"][3]), float(d["score"])]
            if not args.no_gt:
                gtb = match_gt_crop(gt_by_img.get(image_id, []), target, d["box"],
                                    (ox1, oy1), (ch, cw), args.gt_iou)
                if gtb is not None:
                    gt_boxes[stem] = gtb
            if args.dump_emb:
                gh, gw = grid
                emb = tok.detach().cpu().numpy().reshape(gh, gw, -1).transpose(2, 0, 1)
                np.savez_compressed(os.path.join(emb_dir, stem + ".npz"), emb=emb)

            write_facts(os.path.join(args.out_dir, stem + ".lp"), dbx, (ch, cw),
                        R["region"], R["foreigns"], R["region_prims"], R["geoms"],
                        R["boundaries"], R["relations"], R["diag"], cls=args.class_name)
            visualize(crop_rgb, dbx, R["region"], R["support_up"], R["foreigns"],
                      R["region_prims"], R["geoms"], R["boundaries"],
                      os.path.join(args.out_dir, stem + ".png"),
                      title=f"{fname} det#{k} s={d['score']:.2f} | {args.profile} | "
                            f"regions=1+{len(R['foreigns'])} "
                            f"prims={len(R['region_prims'])}+{len(R['geoms'])} "
                            f"boundaries={len(R['boundaries'])} fit_err={R['body']['fit_error']:.2f}")
            n_done += 1
            n_bound += len(R["boundaries"]); n_fo += len(R["foreigns"])
            n_regprim += len(R["region_prims"])
            n_outer += sum(g["location"] == "outer" for g in R["geoms"])
            n_interior += sum(g["location"] == "interior" for g in R["geoms"])
            if n_done >= args.max_dets:
                break
        if n_done >= args.max_dets:
            break

    print(f"\n[{args.profile}] wrote {n_done} decompositions to {args.out_dir}")
    if not args.no_gt:
        gt_out = args.gt_out or os.path.join(args.out_dir, "gt_boxes.json")
        import json as _json
        with open(gt_out, "w") as fh:
            _json.dump(gt_boxes, fh)
        print(f"  gt boxes matched     : {len(gt_boxes)}/{n_done}  -> {gt_out}")
    import json as _json
    cm_out = os.path.join(args.out_dir, "crop_meta.json")
    with open(cm_out, "w") as fh:
        _json.dump(crop_meta, fh)
    print(f"  crop meta written    : {len(crop_meta)}  -> {cm_out}")
    if args.dump_emb:
        print(f"  embeddings dumped    : {n_done}  -> {emb_dir}/<stem>.npz")
    if n_done:
        print(f"  region prims / det   : {n_regprim/n_done:.1f}  (g1 ellipse{' + g2 rect' if 'rect' in prof.region_prims else ''})")
        print(f"  boundaries   / det   : {n_bound/n_done:.1f}  (order {prof.boundary_order})")
        print(f"  PiDiNet frags/ det   : outer {n_outer/n_done:.1f}  interior {n_interior/n_done:.1f}")
        if prof.foreign:
            print(f"  foreign objs / det   : {n_fo/n_done:.2f}")


if __name__ == "__main__":
    main()