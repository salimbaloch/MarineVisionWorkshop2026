#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m2_regions.py  —  Milestone 2: REGION atoms (diagram box 3A -> 3D).

STEP 2 change: the urchin BODY is now a fitted ELLIPSE geometric_form atom
(matching the diagram's "ellipse (body) support/fit_error"), not the old
whole-body 'blob' + concave notches. Each body atom carries BOTH geometry
(centroid, axes, orientation, fit_error) AND the Step-1 DINOv3 semantics
(feature_objectness, region_coherence), so the downstream reasoner sees more
than shape.

Two data-driven decisions from the M1 run are baked in here:
  * objectness is read on the CALIBRATED scale (core.feature_objectness), so a
    clean body reads ~80 instead of ~20.
  * RIDGE-SNAP is implemented but DEFAULT OFF. The hypothesis was that the
    support mask is conservative and should be grown to the B_dino ridge. The
    M2 QA falsified it: ellipse@fit already matches GT (0.74 IoU), and the
    B_dino ridge is the SPINE HALO (outside the body polygon), so snapping
    inflates the body and IoU drops to 0.62. The body is therefore ellipse@fit;
    B_dino's ridge is kept for the Step-4 spike branch, where the halo it marks
    is exactly where the spines live. (--ridge-snap re-enables it for classes
    where the mask genuinely under-segments, e.g. bottle/tire.)

Still exports route_topology / blob_region_atoms / shape_descriptors /
classify_form unchanged (M3 imports them). Notch atoms are available behind
--notches but OFF by default for the ellipse-body representation.

Run on your machine. Deps: torch, numpy, cv2, PIL, scipy, matplotlib, core.py.
"""
import os
import math
import argparse

import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import core


#  descriptors
def shape_descriptors(mask_u8):
    """Full geometric descriptor menu for a binary region (uint8 0/255)."""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c)) or float((mask_u8 > 0).sum())
    perim = float(cv2.arcLength(c, True)) + 1e-6
    M = cv2.moments(c)
    cx = M["m10"] / M["m00"] if M["m00"] else float(c[:, 0, 0].mean())
    cy = M["m01"] / M["m00"] if M["m00"] else float(c[:, 0, 1].mean())
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull)) + 1e-6
    solidity = float(np.clip(area / hull_area, 0, 1))            # = convexity
    circularity = float(np.clip(4 * math.pi * area / (perim * perim), 0, 1))
    x, y, w, h = cv2.boundingRect(c)
    aspect = w / float(h) if h else 1.0
    if len(c) >= 5:
        (_, _), (d1, d2), angle = cv2.fitEllipse(c)
        major, minor = max(d1, d2), max(min(d1, d2), 1e-3)
        orient = angle % 180.0
        eccentricity = float(np.sqrt(max(0.0, 1 - (minor / major) ** 2)))
    else:
        major, minor = max(w, h), max(min(w, h), 1e-3)
        orient = 0.0 if w >= h else 90.0
        eccentricity = 0.0
    elongation = float(major / minor)

    # convexity defects = concave notches between protrusions
    n_defects, mean_depth, max_depth = 0, 0.0, 0.0
    if len(c) >= 4:
        hull_idx = cv2.convexHull(c, returnPoints=False)
        if hull_idx is not None and len(hull_idx) > 3:
            try:
                defs = cv2.convexityDefects(c, hull_idx)
            except cv2.error:
                defs = None
            if defs is not None:
                depths = defs[:, 0, 3] / 256.0                  # px
                eq_r = math.sqrt(area / math.pi) + 1e-6
                sig = depths[depths > 0.06 * eq_r]              # ignore tiny wiggles
                n_defects = int(len(sig))
                if len(sig):
                    mean_depth = float(sig.mean()); max_depth = float(sig.max())
    return dict(area=area, perim=perim, centroid=(cx, cy), bbox=(x, y, x + w, y + h),
                major=float(major), minor=float(minor), elongation=elongation,
                circularity=circularity, solidity=solidity, aspect=float(aspect),
                orient=float(orient), eccentricity=eccentricity,
                n_defects=n_defects, mean_defect_depth=mean_depth, max_defect_depth=max_depth)


def classify_form(d, total_area):
    e, circ, sol = d["elongation"], d["circularity"], d["solidity"]
    small = d["area"] < 0.15 * total_area
    if e > 4.0 and small:
        return "thin_appendage"
    if e > 2.5:
        return "elongated_region"
    if sol < 0.75:
        return "concave_region"
    if circ > 0.78:
        return "ellipse" if e < 1.3 else "blob"
    return "blob"


#  topology router
def radial_profile_peaks(mask_u8, n_bins=72, smooth=2, min_prom_frac=0.12, min_sep_deg=28):
    """Count peaks of the max-radius profile r(theta) around the centroid.
    >=3 peaks suggests radial arms (starfish). Returns (n_peaks, rprofile, arm_prom)."""
    from scipy.signal import find_peaks
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 8:
        return 0, None, 0.0
    cx, cy = xs.mean(), ys.mean()
    ang = np.arctan2(ys - cy, xs - cx)
    rad = np.hypot(xs - cx, ys - cy)
    bins = (((ang + np.pi) / (2 * np.pi)) * n_bins).astype(int) % n_bins
    rprof = np.zeros(n_bins)
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            rprof[b] = rad[sel].max()
    good = rprof > 0
    if 1 < good.sum() < n_bins:
        idx = np.arange(n_bins)
        rprof = np.interp(idx, idx[good], rprof[good], period=n_bins)
    for _ in range(max(0, smooth)):
        rprof = (np.roll(rprof, 1) + rprof + np.roll(rprof, -1)) / 3.0
    prom = min_prom_frac * (rprof.max() - rprof.min() + 1e-6)
    dist = max(1, int(min_sep_deg / 360.0 * n_bins))
    ext = np.concatenate([rprof, rprof, rprof])
    pk, props = find_peaks(ext, prominence=prom, distance=dist)
    peaks = [p - n_bins for p in pk if n_bins <= p < 2 * n_bins]
    arm_prom = 0.0
    if peaks:
        sel = [(n_bins <= p < 2 * n_bins) for p in pk]
        proms = props["prominences"][np.array(sel)] if "prominences" in props else np.array([])
        if proms.size:
            arm_prom = float(proms.mean() / (rprof.mean() + 1e-6))
    return len(peaks), rprof, arm_prom


def has_interior_hole(mask_u8):
    """True if the region has a real interior hole (annulus / tire)."""
    cnts, hier = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    hier = hier[0]
    outer_area = max((cv2.contourArea(c) for c in cnts), default=1.0) + 1e-6
    for i, c in enumerate(cnts):
        if hier[i][3] != -1 and cv2.contourArea(c) > 0.03 * outer_area:   # child of size
            return True
    return False


def route_topology(mask_u8, desc, probe, bimod_thr=2.2,
                   radial_max_solidity=0.70, radial_min_arm_prom=0.30):
    """Decide the decomposition branch. Returns (branch, info).

    Radial (starfish) requires ALL of: 3-8 radius peaks, LOW solidity, and
    sufficient arm prominence. Together they stop urchin/sponge blobs from being
    read as starfish."""
    if probe.get("bimodality", 0.0) >= bimod_thr:
        return "feature_parts", {}                       # multi-material
    if has_interior_hole(mask_u8):
        return "annulus", {}                             # tire
    n_peaks, rprof, arm_prom = radial_profile_peaks(mask_u8)
    if (3 <= n_peaks <= 8
            and desc["solidity"] < radial_max_solidity
            and arm_prom >= radial_min_arm_prom):
        return "radial", dict(n_peaks=n_peaks, arm_prom=round(arm_prom, 2))
    if desc["elongation"] > 2.2 and desc["solidity"] > 0.6:
        return "collinear", {}                           # bottle
    return "blob", {}                                    # urchin / sponge / shell


#  ---------------------------------------------------------- ellipse body (Step 2)
def fit_ellipse_to_mask(mask_u8):
    """cv2.fitEllipse on the largest contour. Returns (cx, cy, d1, d2, angle)
    where d1/d2 are FULL axis lengths (px) and angle is degrees, or None."""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(c)
    return (float(cx), float(cy), float(d1), float(d2), float(ang))


def ellipse_to_mask(shape_hw, ell, scale=1.0):
    """Filled uint8 mask of an ellipse, axes scaled radially by `scale`."""
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    cx, cy, d1, d2, ang = ell
    axes = (max(1, int(round(d1 * scale / 2))), max(1, int(round(d2 * scale / 2))))
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), axes, ang, 0, 360, 255, -1)
    return m


def _ellipse_perim_band(shape_hw, ell, scale, thickness=3):
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    cx, cy, d1, d2, ang = ell
    axes = (max(1, int(round(d1 * scale / 2))), max(1, int(round(d2 * scale / 2))))
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), axes, ang, 0, 360, 255, thickness)
    return m > 0


def snap_ellipse_to_ridge(ell, bdino_up, shape_hw, smin=0.9, smax=1.4, step=0.05,
                          min_gain=1.05):
    """Scale the ellipse radially to land its perimeter on the B_dino ridge.
    Returns (best_scale, energies). Guarded: leaves scale=1.0 unless some other
    scale beats the s=1.0 perimeter energy by at least `min_gain`x (so a mushy /
    flat B_dino on a turbid crop doesn't snap to noise)."""
    energies = {}
    s = smin
    while s <= smax + 1e-9:
        band = _ellipse_perim_band(shape_hw, ell, s)
        energies[round(s, 2)] = float(bdino_up[band].mean()) if band.any() else 0.0
        s += step
    e1 = energies.get(1.0, 0.0)
    best_s = max(energies, key=energies.get)
    best_e = energies[best_s]
    if best_s != 1.0 and best_e < min_gain * (e1 + 1e-6):
        best_s = 1.0                                     # gain too small -> don't snap
    return best_s, energies


def iou_u8(a, b):
    A, B = a > 0, b > 0
    u = np.logical_or(A, B).sum()
    return float(np.logical_and(A, B).sum() / u) if u > 0 else 0.0


def ellipse_body_atom(mask_u8, support, bdino_up, crop_hw, tok, grid, mask_grid,
                      obj_proto, bg_proto, ridge_snap=False, snap_max=1.4):
    """Build the single ELLIPSE body atom: fit to the support mask, ridge-snap to
    the B_dino edge, attach fit_error + Step-1 DINOv3 semantics. Returns an atom
    dict (with .mask = the final ellipse), or None if no ellipse could be fit."""
    ell = fit_ellipse_to_mask(mask_u8)
    if ell is None:
        return None
    ch, cw = crop_hw
    em1 = ellipse_to_mask((ch, cw), ell, 1.0)
    fit_err = 1.0 - iou_u8(em1, mask_u8)                 # ellipticity of the body

    snap_s, snap_energies = (1.0, {})
    if ridge_snap and bdino_up is not None:
        snap_s, snap_energies = snap_ellipse_to_ridge(ell, bdino_up, (ch, cw),
                                                       smax=snap_max)
    em = ellipse_to_mask((ch, cw), ell, snap_s)

    support_up = core.upsample_grid(support, (ch, cw), "bilinear")
    sup = float(support_up[em > 0].mean()) if (em > 0).any() else 0.0

    rfeat = core.region_feature(tok, grid, mask_grid)
    objness = core.feature_objectness(rfeat, obj_proto, bg_proto)   # calibrated
    coher = core.region_coherence(tok, grid, mask_grid)

    desc = shape_descriptors(em)
    cx, cy, d1, d2, ang = ell
    bbox = desc["bbox"] if desc else (0, 0, cw, ch)
    return dict(id="body1", role="body_form", kind="region", form="ellipse",
                centroid=(cx, cy), bbox=bbox, mask=em, desc=desc,
                support=int(round(sup)),
                ellipse=(cx, cy, d1 * snap_s, d2 * snap_s, ang),
                fit_error=fit_err, snap_scale=snap_s, snap_energies=snap_energies,
                objness=objness, coher=coher, rfeat=rfeat)


#  blob branch (urchin notches, kept for --notches / M3 compat)
def blob_region_atoms(mask_u8, support_grid, glabel_full, crop_hw):
    """Whole-body blob atom + concave sub-atoms at the deepest convexity defects."""
    ch, cw = crop_hw
    support_up = cv2.resize(support_grid.astype(np.float32), (cw, ch),
                            interpolation=cv2.INTER_LINEAR)

    def supp_under(binmask):
        v = support_up[binmask > 0]
        return float(v.mean()) if v.size else 0.0

    atoms = []
    d = shape_descriptors(mask_u8)
    if d is None:
        return atoms
    atoms.append(dict(id="body1", role="body_form", kind="region",
                      centroid=d["centroid"], bbox=d["bbox"], mask=mask_u8.copy(),
                      desc=d, form="blob",
                      support=int(round(supp_under(mask_u8)))))

    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if len(c) >= 4:
            hull_idx = cv2.convexHull(c, returnPoints=False)
            if hull_idx is not None and len(hull_idx) > 3:
                try:
                    defs = cv2.convexityDefects(c, hull_idx)
                except cv2.error:
                    defs = None
                if defs is not None:
                    eq_r = math.sqrt(d["area"] / math.pi) + 1e-6
                    order = sorted(range(len(defs)), key=lambda i: -defs[i, 0, 3])
                    taken = 0
                    for i in order:
                        s, e, f, dep = defs[i, 0]
                        depth = dep / 256.0
                        if depth < 0.10 * eq_r:
                            break
                        fx, fy = c[f][0]
                        m = np.zeros_like(mask_u8)
                        cv2.circle(m, (int(fx), int(fy)), max(2, int(0.08 * eq_r)), 255, -1)
                        notch_sup = int(round(supp_under(m)))
                        if notch_sup < 40:
                            continue
                        atoms.append(dict(id=f"notch{taken+1}", role="concave_region",
                                          kind="region", centroid=(float(fx), float(fy)),
                                          bbox=(int(fx)-3, int(fy)-3, int(fx)+3, int(fy)+3),
                                          mask=m, desc=None, form="concave_region",
                                          support=notch_sup,
                                          defect_depth=int(round(depth))))
                        taken += 1
                        if taken >= 6:
                            break
    return atoms


#  facts + viz
def write_facts(path, det_box_crop, branch, info, atoms, oid="o1", cls="object", mq=None):
    L = [f"% region-atom facts (M2, Step 2: ellipse body). branch = {branch}", ""]
    bx = [int(round(v)) for v in det_box_crop]
    L += [f"detection({oid}).", f"class({oid},{cls}).",
          f"proposal_box({oid},box({bx[0]},{bx[1]},{bx[2]},{bx[3]})).",
          f"topology({oid},{branch})."]
    if mq is not None:
        L += [f"mask_contrast({oid},{int(round(mq['contrast']))}).",
              f"mask_frac({oid},{int(round(mq['mask_frac']*100))}).",
              f"boundary_sharpness({oid},{int(round(mq['boundary_sharpness']))})."]
    for a in atoms:
        cx, cy = [int(round(v)) for v in a["centroid"]]
        bb = [int(round(v)) for v in a["bbox"]]
        L += ["", f"atom({a['id']}).", f"belongs_to_detection({a['id']},{oid}).",
              f"atom_kind({a['id']},geometric_form).",
              f"atom_type({a['id']},{a['role']}).",
              f"shape_class({a['id']},{a['form']}).",
              f"centroid({a['id']},point({cx},{cy})).",
              f"atom_box({a['id']},box({bb[0]},{bb[1]},{bb[2]},{bb[3]})).",
              f"support({a['id']},{a['support']}).", f"inside({a['id']},{oid})."]
        # ellipse parameters + fit quality (Step 2)
        if a.get("ellipse") is not None:
            ecx, ecy, ed1, ed2, eang = a["ellipse"]
            L += [f"ellipse({a['id']},point({int(round(ecx))},{int(round(ecy))}),"
                  f"axes({int(round(ed1))},{int(round(ed2))}),angle({int(round(eang))})).",
                  f"fit_error({a['id']},{int(round(a['fit_error']*100))}).",
                  f"snap_scale({a['id']},{int(round(a['snap_scale']*100))})."]
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
                  f"eccentricity({a['id']},{int(round(d['eccentricity']*100))}).",
                  f"aspect_ratio({a['id']},{int(round(d['aspect']*100))}).",
                  f"orientation({a['id']},{int(round(d['orient']))}).",
                  f"convexity_defects({a['id']},{d['n_defects']}).",
                  f"mean_defect_depth({a['id']},{int(round(d['mean_defect_depth']))})."]
        if "defect_depth" in a:
            L.append(f"defect_depth({a['id']},{a['defect_depth']}).")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def visualize(crop_rgb, det_box_crop, mask_u8, bdino_up, body, atoms, out_path, title):
    H, W = crop_rgb.shape[:2]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    bx1, by1, bx2, by2 = det_box_crop

    ax[0].imshow(crop_rgb)
    ax[0].add_patch(Rectangle((bx1, by1), bx2-bx1, by2-by1, fill=False, ec="lime", lw=2))
    ov = np.zeros((H, W, 4)); ov[mask_u8 > 0] = [1, 1, 1, 0.25]; ax[0].imshow(ov)
    ax[0].set_title("crop + DETR box + support mask")

    # mask (white fill) + ellipse@1.0 (yellow) + ellipse@snap (lime)
    ax[1].imshow(crop_rgb)
    ax[1].imshow(ov)
    if body is not None:
        ell = (body["ellipse"][0], body["ellipse"][1],
               body["ellipse"][2] / body["snap_scale"],
               body["ellipse"][3] / body["snap_scale"], body["ellipse"][4])
        e1 = ellipse_to_mask((H, W), ell, 1.0)
        es = body["mask"]
        ax[1].contour(e1.astype(float), levels=[127], colors="yellow", linewidths=1.4)
        ax[1].contour(es.astype(float), levels=[127], colors="lime", linewidths=1.6)
        ax[1].set_title(f"ellipse: fit (yellow) -> snap x{body['snap_scale']:.2f} (lime)  "
                        f"fit_err={body['fit_error']:.2f}")
    else:
        ax[1].set_title("no ellipse")

    ax[2].imshow(crop_rgb)
    if bdino_up is not None:
        ax[2].imshow(bdino_up / (bdino_up.max() + 1e-6), cmap="magma", alpha=0.6)
    if body is not None:
        ax[2].contour(body["mask"].astype(float), levels=[127], colors="cyan", linewidths=1.6)
    ax[2].set_title("snapped ellipse (cyan) on B_dino ridge")

    for a in ax:
        a.set_xlim(0, W); a.set_ylim(H, 0); a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


#  QA helpers
def poly_to_mask(seg, h, w):
    from pycocotools import mask as coco_mask
    if not isinstance(seg, list):
        return None
    rles = coco_mask.frPyObjects(seg, h, w)
    rle = coco_mask.merge(rles) if isinstance(rles, list) else rles
    return coco_mask.decode(rle).astype(np.uint8)


def _place_full(mask_crop, full_hw, off):
    H, W = full_hw; ox, oy = off
    ch, cw = mask_crop.shape[:2]
    full = np.zeros((H, W), bool)
    full[oy:oy+ch, ox:ox+cw] = mask_crop > 0
    return full


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


#  main
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
    ap.add_argument("--conn", type=int, choices=[4, 8], default=8)
    ap.add_argument("--ridge-snap", dest="ridge_snap", action="store_true", default=False,
                    help="snap body ellipse out to the B_dino ridge. DEFAULT OFF: QA showed "
                         "the ridge is the spine halo, not the body edge, so snapping inflates "
                         "the body past GT (0.62 vs 0.74 IoU). ellipse@fit is the body.")
    ap.add_argument("--no-ridge-snap", dest="ridge_snap", action="store_false")
    ap.add_argument("--snap-max", type=float, default=1.4)
    ap.add_argument("--notches", action="store_true", help="also emit concave notch atoms")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--min-grid", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--max-dets", type=int, default=40)
    ap.add_argument("--qa", action="store_true", help="3-way mask-IoU vs GT polygon; no png/lp")
    ap.add_argument("--match-thr", type=float, default=0.1)
    ap.add_argument("--out-dir", default="./m2_out")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.fresh and os.path.isdir(args.out_dir):
        import shutil; shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    id2file, name2cat, cat2name, gt_seg = core.load_gt_index(args.gt_json)
    if args.class_name not in name2cat:
        raise SystemExit(f"'{args.class_name}' not in {list(name2cat)[:12]}...")
    target = name2cat[args.class_name]
    print(f"Target '{args.class_name}' -> id={target}")
    model = core.load_dinov3(args.repo_root, args.weights, args.weights_kind,
                             prefer_ema=not args.no_ema, device=device)
    by_img = core.load_predictions(args.pred_json, args.box_format)

    import json
    id2hw = {im["id"]: (im["height"], im["width"])
             for im in json.load(open(args.gt_json))["images"]} if args.qa else {}

    branch_counts = {}
    qa_rows = []
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
                print(f"[skip] {stem}: empty mask (gate={gate:.2f})")
                continue

            probe = core.feature_variance_probe(tok, grid, mask_grid)
            mq = core.mask_quality(support, mask_grid)
            bdino = core.feature_boundary_map(tok, grid, connectivity=args.conn)
            bdino_up = core.upsample_grid(bdino, (ch, cw), "bilinear")
            obj_proto, bg_proto = core.object_bg_prototypes(
                tok, grid, mhw, dbx, (ch, cw), args.seed_frac)
            mask_u8 = (core.upsample_mask(mask_grid, (ch, cw)).astype(np.uint8)) * 255
            desc0 = shape_descriptors(mask_u8)
            if desc0 is None:
                continue
            branch, info = route_topology(mask_u8, desc0, probe, args.bimod_thr)
            branch_counts[branch] = branch_counts.get(branch, 0) + 1

            body = ellipse_body_atom(mask_u8, support, bdino_up, (ch, cw), tok, grid,
                                     mask_grid, obj_proto, bg_proto,
                                     ridge_snap=args.ridge_snap, snap_max=args.snap_max)
            if body is None:
                print(f"[skip] {stem}: ellipse fit failed")
                continue
            atoms = [body]
            if args.notches and branch == "blob":
                notches = [a for a in blob_region_atoms(mask_u8, support, mask_grid, (ch, cw))
                           if a["role"] == "concave_region"]
                atoms += notches

            if args.qa:
                bj, best = -1, args.match_thr
                for j, g in enumerate(gts):
                    if used[j]:
                        continue
                    v = _box_iou(d["box"], g["box"])
                    if v >= best:
                        best, bj = v, j
                if bj < 0:
                    done += 1
                    continue
                used[bj] = True
                H, W = id2hw[image_id]
                gmask = poly_to_mask(gts[bj]["seg"], H, W)
                if gmask is not None:
                    off = (ox1, oy1)
                    f_mask = _place_full(mask_u8, (H, W), off)
                    f_ell1 = _place_full(ellipse_to_mask((ch, cw),
                                         fit_ellipse_to_mask(mask_u8), 1.0), (H, W), off)
                    f_ells = _place_full(body["mask"], (H, W), off)
                    g = gmask > 0
                    qa_rows.append(dict(
                        miou_mask=iou_u8(f_mask, g), miou_ell1=iou_u8(f_ell1, g),
                        miou_snap=iou_u8(f_ells, g), fit_error=body["fit_error"],
                        snap=body["snap_scale"], objness=body["objness"],
                        coher=body["coher"]))
            else:
                write_facts(os.path.join(args.out_dir, stem + ".lp"), dbx, branch, info,
                            atoms, cls=args.class_name, mq=mq)
                visualize(crop_rgb, dbx, mask_u8, bdino_up, body, atoms,
                          os.path.join(args.out_dir, stem + ".png"),
                          title=f"{fname} det#{k} s={d['score']:.2f} branch={branch} "
                                f"fit_err={body['fit_error']:.2f} snap=x{body['snap_scale']:.2f} "
                                f"objness={body['objness']:.0f} coher={body['coher']:.0f}")
            done += 1
            if done >= args.max_dets:
                break
        if done >= args.max_dets:
            break

    if args.qa:
        qa_report(qa_rows)
    else:
        print(f"\nWrote {done} detections to {args.out_dir}")
    print(f"router branch counts: {branch_counts}")


def qa_report(rows):
    if not rows:
        print("No matched detections."); return
    a = lambda k: np.array([r[k] for r in rows], float)
    mm, e1, es = a("miou_mask"), a("miou_ell1"), a("miou_snap")
    print(f"\n===== M2 ELLIPSE-BODY QA  ({len(rows)} matched detections) =====")
    print(f"  mask-IoU vs GT polygon (higher = better):")
    print(f"    raw support mask     mean {mm.mean():.3f}  median {np.median(mm):.3f}")
    print(f"    ellipse @ fit        mean {e1.mean():.3f}  median {np.median(e1):.3f}")
    print(f"    ellipse @ ridge-snap mean {es.mean():.3f}  median {np.median(es):.3f}")
    print(f"\n    ellipse@fit beats raw mask  : {100*(e1 > mm).mean():.0f}% of dets")
    print(f"    ridge-snap beats ellipse@fit: {100*(es > e1).mean():.0f}% of dets")
    print(f"    ridge-snap beats raw mask   : {100*(es > mm).mean():.0f}% of dets")
    print(f"\n  fit_error (1-IoU ellipse vs mask): mean {a('fit_error').mean():.3f}  "
          f"median {np.median(a('fit_error')):.3f}   (low = body really is elliptical)")
    print(f"  snap scale: mean x{a('snap').mean():.3f}  median x{np.median(a('snap')):.3f}  "
          f"(>1 = mask was conservative, snap grew it to the ridge)")
    print(f"  feature_objectness (calibrated): mean {a('objness').mean():.1f}  "
          f"median {np.median(a('objness')):.1f}   (should now read high on bodies)")
    print(f"  region_coherence: mean {a('coher').mean():.1f}")
    print("\n  read: if ellipse@fit ~ raw mask, the body IS elliptical (good, the")
    print("  ellipse loses nothing). If ridge-snap > both, the conservative support")
    print("  mask was the bottleneck and B_dino fixed the localization.")


if __name__ == "__main__":
    main()