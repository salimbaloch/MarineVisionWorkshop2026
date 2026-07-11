#!/usr/bin/env python3
import os
import json
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# -----------------------------
# Basic geometry helpers
# -----------------------------
def xywh_to_xyxy(b):
    x, y, w, h = map(float, b)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = map(float, b)
    return np.array([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float32)


def clip_xyxy(b, W, H):
    x1, y1, x2, y2 = map(float, b)
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 + 1:
        x2 = min(W, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(H, y1 + 2)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def expand_xyxy(b, scale, W, H):
    x1, y1, x2, y2 = map(float, b)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return clip_xyxy([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], W, H)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_area_xyxy(b):
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area_xyxy(a) + box_area_xyxy(b) - inter
    return inter / union if union > 0 else 0.0


def local_to_global_box(b, crop_xyxy):
    x1, y1, x2, y2 = map(float, b)
    cx1, cy1, _, _ = map(float, crop_xyxy)
    return np.array([x1 + cx1, y1 + cy1, x2 + cx1, y2 + cy1], dtype=np.float32)


def safe_name(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


# -----------------------------
# COCO / prediction helpers
# -----------------------------
def load_categories(gt_json):
    data = json.load(open(gt_json, "r"))
    cats = data.get("categories", [])
    name_to_id = {c["name"]: c["id"] for c in cats}
    return data, name_to_id


def image_lookup(gt_data):
    out = {}
    for im in gt_data.get("images", []):
        out[int(im["id"])] = im
    return out


def load_predictions(pred_json, cat_id, score_thr, max_dets):
    preds = json.load(open(pred_json, "r"))
    rows = []
    for p in preds:
        if int(p.get("category_id", -999)) != int(cat_id):
            continue
        if float(p.get("score", 0.0)) < score_thr:
            continue
        rows.append(p)
    rows.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    if max_dets > 0:
        rows = rows[:max_dets]
    return rows


def gt_by_image_cat(gt_data, cat_id):
    out = {}
    for ann in gt_data.get("annotations", []):
        if int(ann.get("category_id", -999)) != int(cat_id):
            continue
        out.setdefault(int(ann["image_id"]), []).append(ann)
    return out


def best_gt_for_box(gt_anns, box_xyxy):
    best = None
    best_iou = 0.0
    for ann in gt_anns:
        gt = xywh_to_xyxy(ann["bbox"])
        iou = box_iou_xyxy(box_xyxy, gt)
        if iou > best_iou:
            best_iou = iou
            best = ann
    return best, best_iou


# -----------------------------
# Visual evidence maps
# -----------------------------
def grabcut_support(crop_rgb, det_rect_local):
    """
    Class-agnostic soft support from GrabCut initialized by a soft mask.

    Important V2 change:
    - The original detector box is NOT used as a hard foreground rectangle.
    - The expanded crop outside the detector box is allowed to become foreground.
    - Only the crop border is hard background.
    This makes recovery of missed object parts possible.
    """
    h, w = crop_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in det_rect_local]
    x1 = max(1, min(w - 2, x1))
    y1 = max(1, min(h - 2, y1))
    x2 = max(x1 + 2, min(w - 1, x2))
    y2 = max(y1 + 2, min(h - 1, y2))

    img_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    # Start as probable background, not hard background.
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    # Crop border is hard background.
    border = max(3, int(0.03 * min(h, w)))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    # Slightly expanded detector area is probable foreground.
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    bw, bh = x2 - x1, y2 - y1
    ex1 = int(max(1, cx - 0.65 * bw))
    ey1 = int(max(1, cy - 0.65 * bh))
    ex2 = int(min(w - 1, cx + 0.65 * bw))
    ey2 = int(min(h - 1, cy + 0.65 * bh))
    mask[ey1:ey2, ex1:ex2] = cv2.GC_PR_FGD

    # Inner detector core is sure foreground.
    ix1 = int(max(1, cx - 0.30 * bw))
    iy1 = int(max(1, cy - 0.30 * bh))
    ix2 = int(min(w - 1, cx + 0.30 * bw))
    iy2 = int(min(h - 1, cy + 0.30 * bh))
    mask[iy1:iy2, ix1:ix2] = cv2.GC_FGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    except Exception:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sx, sy = max(8.0, 0.8 * bw), max(8.0, 0.8 * bh)
        fg = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy))).astype(np.float32)

    # Smooth and normalize for atom extraction.
    fg = cv2.GaussianBlur(fg, (0, 0), 4.0)
    fg = cv2.normalize(fg, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return fg.astype(np.float32)


def edge_map(crop_rgb):
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    med = np.median(gray)
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi)
    return edges


def smooth_binary(mask, k=5):
    m = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((k, k), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return (m > 0).astype(np.uint8)


def largest_cc(mask):
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (lab == idx).astype(np.uint8)


def contour_smooth_mask(mask, eps_ratio=0.006):
    """
    Return sm (oothed contour mask and contours.
    """
    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    smoothed = []
    for c in contours:
        if len(c) < 8:
            continue
        peri = cv2.arcLength(c, True)
        eps = max(1.0, eps_ratio * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        cv2.drawContours(out, [approx], -1, 255, thickness=-1)
        smoothed.append(approx)
    return (out > 0).astype(np.uint8), smoothed


def compactness(mask):
    m = (mask > 0).astype(np.uint8) * 255
    area = float((m > 0).sum())
    if area <= 1:
        return 0.0
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    peri = sum(cv2.arcLength(c, True) for c in contours)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (peri * peri))


def rectangularity(mask):
    b = bbox_from_mask(mask)
    if b is None:
        return 0.0
    area = float((mask > 0).sum())
    ba = box_area_xyxy(b)
    return float(area / ba) if ba > 0 else 0.0


def score_bin(v):
    v = float(v)
    if v >= 0.75:
        return "high"
    if v >= 0.45:
        return "medium"
    return "low"


def int100(v):
    return int(round(max(0.0, min(1.0, float(v))) * 100))


# -----------------------------
# Atom extraction
# -----------------------------
def make_region_atoms(support, thresholds=(0.45, 0.55, 0.65, 0.75)):
    atoms = []
    h, w = support.shape
    crop_area = float(h * w)

    for t in thresholds:
        raw = (support >= t).astype(np.uint8)
        raw = smooth_binary(raw, 5)
        cc = largest_cc(raw)
        sm, contours = contour_smooth_mask(cc)

        b = bbox_from_mask(sm)
        if b is None:
            continue

        area_ratio = float(sm.sum()) / max(1.0, crop_area)
        sup_mean = float(support[sm > 0].mean()) if sm.sum() > 0 else 0.0
        sup_max = float(support[sm > 0].max()) if sm.sum() > 0 else 0.0

        atoms.append({
            "kind": "semantic_region",
            "role": "body",
            "name": f"r_p{int(t*100)}",
            "threshold": t,
            "mask": sm,
            "box": b,
            "support_mean": sup_mean,
            "support_max": sup_max,
            "area_ratio": area_ratio,
            "compactness": compactness(sm),
            "rectangularity": rectangularity(sm),
            "contours": contours,
        })

    # multi-component low threshold union: useful for fragmented / translucent objects
    raw = (support >= 0.35).astype(np.uint8)
    raw = smooth_binary(raw, 3)
    b = bbox_from_mask(raw)
    if b is not None:
        atoms.append({
            "kind": "semantic_region",
            "role": "body",
            "name": "r_multi_p35",
            "threshold": 0.35,
            "mask": raw,
            "box": b,
            "support_mean": float(support[raw > 0].mean()) if raw.sum() > 0 else 0.0,
            "support_max": float(support[raw > 0].max()) if raw.sum() > 0 else 0.0,
            "area_ratio": float(raw.sum()) / max(1.0, crop_area),
            "compactness": compactness(raw),
            "rectangularity": rectangularity(raw),
            "contours": [],
        })

    return atoms


def make_perimeter_atoms(region_atoms):
    atoms = []
    for r in region_atoms:
        m = r["mask"].astype(np.uint8)
        if m.sum() < 10:
            continue
        kernel = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(m, kernel, iterations=1)
        ero = cv2.erode(m, kernel, iterations=1)
        per = ((dil - ero) > 0).astype(np.uint8)
        b = bbox_from_mask(per)
        if b is None:
            continue
        atoms.append({
            "kind": "support_perimeter",
            "name": "p_" + r["name"],
            "from_region": r["name"],
            "mask": per,
            "box": b,
            "smoothness": r["compactness"],
        })
    return atoms


def make_geometric_atoms(region_atoms):
    atoms = []

    for r in region_atoms:
        m = (r["mask"] > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 10:
            continue

        # convex hull
        hull = cv2.convexHull(c)
        hm = np.zeros_like(m)
        cv2.drawContours(hm, [hull], -1, 255, thickness=-1)
        hb = bbox_from_mask(hm)
        if hb is not None:
            atoms.append({
                "kind": "geometric_primitive",
                "type": "convex_hull",
                "name": "g_hull_" + r["name"],
                "from_region": r["name"],
                "mask": (hm > 0).astype(np.uint8),
                "box": hb,
                "fit_error": 1.0 - min(1.0, float(r["mask"].sum()) / max(1.0, float((hm > 0).sum()))),
            })

        # min area rectangle
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.int32)
        rm = np.zeros_like(m)
        cv2.drawContours(rm, [box], -1, 255, thickness=-1)
        rb = bbox_from_mask(rm)
        if rb is not None:
            atoms.append({
                "kind": "geometric_primitive",
                "type": "min_rect",
                "name": "g_rect_" + r["name"],
                "from_region": r["name"],
                "mask": (rm > 0).astype(np.uint8),
                "box": rb,
                "fit_error": 1.0 - min(1.0, float(r["mask"].sum()) / max(1.0, float((rm > 0).sum()))),
            })

        # ellipse if enough contour points
        if len(c) >= 5:
            try:
                ellipse = cv2.fitEllipse(c)
                em = np.zeros_like(m)
                cv2.ellipse(em, ellipse, 255, thickness=-1)
                eb = bbox_from_mask(em)
                if eb is not None:
                    atoms.append({
                        "kind": "geometric_primitive",
                        "type": "ellipse",
                        "name": "g_ellipse_" + r["name"],
                        "from_region": r["name"],
                        "mask": (em > 0).astype(np.uint8),
                        "box": eb,
                        "fit_error": 1.0 - min(1.0, float(r["mask"].sum()) / max(1.0, float((em > 0).sum()))),
                    })
            except Exception:
                pass

    return atoms


def make_boundary_atoms(edges, support, perimeter_atoms, min_len=12):
    atoms = []
    h, w = support.shape

    per_union = np.zeros((h, w), np.uint8)
    for p in perimeter_atoms:
        per_union |= p["mask"].astype(np.uint8)

    # keep edges near support perimeter
    band = cv2.dilate(per_union, np.ones((9, 9), np.uint8), iterations=1)
    valid = ((edges > 0) & (band > 0)).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(valid, 8)
    idx = 0
    for cc in range(1, n):
        area = int(stats[cc, cv2.CC_STAT_AREA])
        if area < min_len:
            continue

        x = int(stats[cc, cv2.CC_STAT_LEFT])
        y = int(stats[cc, cv2.CC_STAT_TOP])
        ww = int(stats[cc, cv2.CC_STAT_WIDTH])
        hh = int(stats[cc, cv2.CC_STAT_HEIGHT])

        comp = (lab == cc).astype(np.uint8)
        b = bbox_from_mask(comp)
        if b is None:
            continue

        # approximate side/interior test using support values around component
        dil = cv2.dilate(comp, np.ones((7, 7), np.uint8), iterations=1)
        ring = ((dil > 0) & (comp == 0))
        near_vals = support[ring] if ring.sum() > 0 else np.array([0.0])
        edge_vals = support[comp > 0] if comp.sum() > 0 else np.array([0.0])

        support_edge = float(edge_vals.mean())
        support_ring = float(near_vals.mean())
        side_contrast = abs(support_edge - support_ring)

        # perimeter overlap
        per_overlap = float(((comp > 0) & (band > 0)).sum()) / max(1.0, float(comp.sum()))

        # reject obvious tiny box noise
        if max(ww, hh) < min_len:
            continue

        idx += 1
        atoms.append({
            "kind": "boundary_fragment",
            "name": f"b{idx}",
            "mask": comp,
            "box": b,
            "edge_len": area,
            "edge_confidence": min(1.0, area / 80.0),
            "side_contrast": float(side_contrast),
            "near_perimeter": per_overlap,
            "role": "outer_candidate" if per_overlap > 0.4 else "unknown",
        })

    return atoms


def make_thin_extension_atoms(edges, support, region_atoms, min_len=15):
    atoms = []
    if not region_atoms:
        return atoms

    # Use strongest body region as main body
    main = max(region_atoms, key=lambda r: r["support_mean"])
    body = main["mask"].astype(np.uint8)

    low = (support >= 0.28).astype(np.uint8)
    body_dil = cv2.dilate(body, np.ones((11, 11), np.uint8), iterations=1)

    # thin evidence: low support or edges near but outside the main body
    candidate = (((low > 0) | (edges > 0)) & (body == 0))
    near_body = cv2.dilate(body, np.ones((25, 25), np.uint8), iterations=1)
    candidate = (candidate & (near_body > 0)).astype(np.uint8)

    # remove dense blobs; keep elongated components
    n, lab, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    idx = 0
    for cc in range(1, n):
        area = int(stats[cc, cv2.CC_STAT_AREA])
        if area < min_len:
            continue

        x = int(stats[cc, cv2.CC_STAT_LEFT])
        y = int(stats[cc, cv2.CC_STAT_TOP])
        ww = int(stats[cc, cv2.CC_STAT_WIDTH])
        hh = int(stats[cc, cv2.CC_STAT_HEIGHT])
        short = max(1, min(ww, hh))
        long = max(ww, hh)
        elong = long / short
        fill = area / max(1.0, ww * hh)

        # thin/elongated/fragmented structures
        if elong < 2.0 and fill > 0.45:
            continue

        comp = (lab == cc).astype(np.uint8)
        # must be attached/near body dilation
        attach = ((cv2.dilate(comp, np.ones((5, 5), np.uint8), iterations=1) > 0) & (body_dil > 0)).sum()
        if attach <= 0:
            continue

        b = bbox_from_mask(comp)
        if b is None:
            continue

        idx += 1
        atoms.append({
            "kind": "thin_extension",
            "name": f"t{idx}",
            "attached_to": main["name"],
            "mask": comp,
            "box": b,
            "length": long,
            "width": short,
            "elongation": float(elong),
            "fill": float(fill),
            "support_mean": float(support[comp > 0].mean()) if comp.sum() > 0 else 0.0,
            "edge_confidence": min(1.0, float((edges[comp > 0] > 0).sum()) / max(1.0, area)),
        })

    return atoms


# -----------------------------
# LP fact writing
# -----------------------------
def write_lp(path, stem, cls_name, det_score, det_box_global, crop_box_global, atoms_global):
    def q(s):
        return str(s).replace("-", "_").replace(".", "_")

    lines = []
    O = "o1"
    lines.append(f"detection({O}).")
    lines.append(f"class({O},{q(cls_name)}).")
    lines.append(f"score({O},{int100(det_score)}).")

    dx, dy, dw, dh = xyxy_to_xywh(det_box_global)
    cx, cy, cw, ch = xyxy_to_xywh(crop_box_global)
    lines.append(f"det_box({O},{int(round(dx))},{int(round(dy))},{int(round(dw))},{int(round(dh))}).")
    lines.append(f"crop_box({O},{int(round(cx))},{int(round(cy))},{int(round(cw))},{int(round(ch))}).")

    for a in atoms_global:
        name = q(a["name"])
        kind = a["kind"]
        bx, by, bw, bh = xyxy_to_xywh(a["box_global"])

        lines.append("")
        lines.append(f"candidate({name},{O}).")
        lines.append(f"bbox({name},{int(round(bx))},{int(round(by))},{int(round(bw))},{int(round(bh))}).")

        if kind == "semantic_region":
            lines.append(f"semantic_region({name}).")
            lines.append(f"region_role({name},body).")
            lines.append(f"support_mean({name},{int100(a.get('support_mean',0))}).")
            lines.append(f"support_max({name},{int100(a.get('support_max',0))}).")
            lines.append(f"area_ratio({name},{int100(a.get('area_ratio',0))}).")
            lines.append(f"compactness({name},{int100(a.get('compactness',0))}).")
            lines.append(f"rectangularity({name},{int100(a.get('rectangularity',0))}).")
            lines.append(f"threshold({name},{int(round(100*a.get('threshold',0)))}).")

        elif kind == "support_perimeter":
            lines.append(f"support_perimeter({name}).")
            lines.append(f"from_region({name},{q(a.get('from_region','none'))}).")
            lines.append(f"perimeter_smoothness({name},{int100(a.get('smoothness',0))}).")

        elif kind == "geometric_primitive":
            lines.append(f"geometric_primitive({name}).")
            lines.append(f"primitive_type({name},{q(a.get('type','unknown'))}).")
            lines.append(f"from_region({name},{q(a.get('from_region','none'))}).")
            lines.append(f"fit_error({name},{int100(a.get('fit_error',0))}).")

        elif kind == "boundary_fragment":
            lines.append(f"boundary_fragment({name}).")
            lines.append(f"boundary_role({name},{q(a.get('role','unknown'))}).")
            lines.append(f"edge_confidence({name},{int100(a.get('edge_confidence',0))}).")
            lines.append(f"side_contrast({name},{int100(a.get('side_contrast',0))}).")
            lines.append(f"near_support_perimeter({name},{int100(a.get('near_perimeter',0))}).")
            lines.append(f"edge_length({name},{int(round(a.get('edge_len',0)))}).")

        elif kind == "thin_extension":
            lines.append(f"thin_extension({name}).")
            lines.append(f"attached_to({name},{q(a.get('attached_to','none'))}).")
            lines.append(f"length_px({name},{int(round(a.get('length',0)))}).")
            lines.append(f"width_px({name},{int(round(a.get('width',0)))}).")
            lines.append(f"elongation({name},{int100(min(1.0, a.get('elongation',0)/10.0))}).")
            lines.append(f"support_mean({name},{int100(a.get('support_mean',0))}).")
            lines.append(f"edge_confidence({name},{int100(a.get('edge_confidence',0))}).")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------
# Visualization
# -----------------------------
def color_mask(rgb, mask, color, alpha=0.45):
    out = rgb.copy().astype(np.float32)
    c = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_box(img, box, color, thickness=2, label=None):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def panel_title(img, title):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (255, 255, 255), -1)
    cv2.putText(out, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def resize_panel(img, size=(320, 240)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def make_viz(crop_rgb, det_local, support, regions, perimeters, boundaries, thin_exts, geoms, out_path):
    h, w = crop_rgb.shape[:2]

    p0 = crop_rgb.copy()
    draw_box(p0, det_local, (0, 150, 255), 2, "det")
    p0 = panel_title(p0, "RGB crop + detector box")

    heat = (support * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    p1 = panel_title(heat, "soft support map")

    p2 = crop_rgb.copy()
    for r in regions:
        p2 = color_mask(p2, r["mask"], (255, 230, 0), 0.35)
        draw_box(p2, r["box"], (255, 210, 0), 1, r["name"])
    p2 = panel_title(p2, "smooth support-region atoms")

    p3 = crop_rgb.copy()
    for p in perimeters:
        yy, xx = np.where(p["mask"] > 0)
        p3[yy, xx] = (255, 0, 255)
    p3 = panel_title(p3, "support perimeter atoms")

    p4 = crop_rgb.copy()
    for b in boundaries:
        yy, xx = np.where(b["mask"] > 0)
        p4[yy, xx] = (255, 140, 0)
        draw_box(p4, b["box"], (255, 140, 0), 1)
    p4 = panel_title(p4, "outer-boundary fragments")

    p5 = crop_rgb.copy()
    for t in thin_exts:
        p5 = color_mask(p5, t["mask"], (0, 255, 255), 0.55)
        draw_box(p5, t["box"], (0, 255, 255), 1, t["name"])
    p5 = panel_title(p5, "thin-extension atoms")

    p6 = crop_rgb.copy()
    for g in geoms[:12]:
        draw_box(p6, g["box"], (0, 255, 0), 1, g.get("type", "geom"))
    p6 = panel_title(p6, "geometric primitive boxes")

    p7 = crop_rgb.copy()
    for r in regions[:4]:
        draw_box(p7, r["box"], (255, 210, 0), 2, r["name"])
    for t in thin_exts[:8]:
        draw_box(p7, t["box"], (0, 255, 255), 1, t["name"])
    for b in boundaries[:8]:
        draw_box(p7, b["box"], (255, 140, 0), 1, b["name"])
    draw_box(p7, det_local, (0, 150, 255), 2, "det")
    p7 = panel_title(p7, "all V2 atom boxes")

    panels = [p0, p1, p2, p3, p4, p5, p6, p7]
    panels = [resize_panel(p, (320, 240)) for p in panels]
    row1 = np.concatenate(panels[:4], axis=1)
    row2 = np.concatenate(panels[4:], axis=1)
    canvas = np.concatenate([row1, row2], axis=0)
    Image.fromarray(canvas).save(out_path)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("decomposev2.py -- class-agnostic V2 visual atom extractor")
    ap.add_argument("--pred-json", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--class-name", required=True)
    ap.add_argument("--score-thr", type=float, default=0.2)
    ap.add_argument("--max-dets", type=int, default=20)
    ap.add_argument("--crop-scale", type=float, default=1.5)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--box-format", default="xywh", choices=["xywh"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    lp_dir = out_dir / "lp"
    viz_dir = out_dir / "viz"
    meta_path = out_dir / "crop_meta.json"
    gtbox_path = out_dir / "gt_boxes.json"
    lp_dir.mkdir(parents=True, exist_ok=True)
    if args.viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    gt_data, name_to_id = load_categories(args.gt_json)
    if args.class_name not in name_to_id:
        raise SystemExit(f"class not found in GT categories: {args.class_name}")

    cat_id = name_to_id[args.class_name]
    imgs = image_lookup(gt_data)
    preds = load_predictions(args.pred_json, cat_id, args.score_thr, args.max_dets)
    gtb = gt_by_image_cat(gt_data, cat_id)

    print(f"[v2] class={args.class_name} cat_id={cat_id} preds={len(preds)} out={out_dir}")

    crop_meta = []
    gt_boxes = {}
    written = 0

    for i, p in enumerate(preds):
        img_id = int(p["image_id"])
        if img_id not in imgs:
            continue
        info = imgs[img_id]
        fn = info["file_name"]
        img_path = os.path.join(args.img_root, fn)
        if not os.path.exists(img_path):
            # try basename fallback
            img_path = os.path.join(args.img_root, os.path.basename(fn))
        if not os.path.exists(img_path):
            print(f"[skip] image missing: {fn}")
            continue

        rgb = np.array(Image.open(img_path).convert("RGB"))
        H, W = rgb.shape[:2]

        det_xyxy = clip_xyxy(xywh_to_xyxy(p["bbox"]), W, H)
        crop_xyxy = expand_xyxy(det_xyxy, args.crop_scale, W, H)

        x1, y1, x2, y2 = [int(round(v)) for v in crop_xyxy]
        crop = rgb[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        det_local = det_xyxy.copy()
        det_local[[0, 2]] -= x1
        det_local[[1, 3]] -= y1
        det_local = clip_xyxy(det_local, crop.shape[1], crop.shape[0])

        support = grabcut_support(crop, det_local)
        edges = edge_map(crop)

        regions = make_region_atoms(support)
        perimeters = make_perimeter_atoms(regions)
        geoms = make_geometric_atoms(regions)
        boundaries = make_boundary_atoms(edges, support, perimeters)
        thin_exts = make_thin_extension_atoms(edges, support, regions)

        atoms = []
        atoms.extend(regions)
        atoms.extend(perimeters)
        atoms.extend(geoms)
        atoms.extend(boundaries)
        atoms.extend(thin_exts)

        atoms_global = []
        for a in atoms:
            ag = dict(a)
            ag["box_global"] = local_to_global_box(a["box"], crop_xyxy)
            # Remove masks from facts/meta to avoid bloated json
            atoms_global.append(ag)

        stem = f"{Path(fn).stem}_det{i}"
        stem = safe_name(stem)

        lp_path = lp_dir / f"{stem}.lp"
        write_lp(lp_path, stem, args.class_name, float(p.get("score", 0.0)), det_xyxy, crop_xyxy, atoms_global)

        if args.viz:
            make_viz(crop, det_local, support, regions, perimeters, boundaries, thin_exts, geoms, viz_dir / f"{stem}.png")

        best_gt, biou = best_gt_for_box(gtb.get(img_id, []), det_xyxy)
        if best_gt is not None:
            gt_boxes[stem] = {
                "bbox": best_gt["bbox"],
                "iou_det_gt": biou,
                "ann_id": best_gt.get("id", -1),
            }

        crop_meta.append({
            "stem": stem,
            "image_id": img_id,
            "file_name": fn,
            "category_id": cat_id,
            "class_name": args.class_name,
            "score": float(p.get("score", 0.0)),
            "det_box_xywh": xyxy_to_xywh(det_xyxy).tolist(),
            "crop_box_xywh": xyxy_to_xywh(crop_xyxy).tolist(),
            "n_regions": len(regions),
            "n_perimeters": len(perimeters),
            "n_geoms": len(geoms),
            "n_boundaries": len(boundaries),
            "n_thin_exts": len(thin_exts),
            "lp": str(lp_path),
            "viz": str(viz_dir / f"{stem}.png") if args.viz else "",
        })

        written += 1
        print(
            f"[{written:04d}] {stem}: "
            f"regions={len(regions)} perim={len(perimeters)} geom={len(geoms)} "
            f"bnd={len(boundaries)} thin={len(thin_exts)} detIoU={biou:.3f}"
        )

    json.dump(crop_meta, open(meta_path, "w"), indent=2)
    json.dump(gt_boxes, open(gtbox_path, "w"), indent=2)

    print("")
    print(f"[v2] wrote {written} decompositions")
    print(f"     lp       : {lp_dir}")
    if args.viz:
        print(f"     viz      : {viz_dir}")
    print(f"     crop_meta: {meta_path}")
    print(f"     gt_boxes : {gtbox_path}")


if __name__ == "__main__":
    main()
