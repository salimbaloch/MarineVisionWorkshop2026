#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core.py  —  shared substrate for the atom-generation pipeline (M1/M2/M3).

Consolidates the machinery that was copy-pasted three times across the old
M1/M2/M3:
  * DINOv3 (ViT-S/16, DEIMv2-L backbone) checkpoint loading + token extraction
  * crop_with_margin / to_model_input
  * compute_support      (the proven v2 support map)
  * clean_mask           (threshold -> largest-CC -> fill-holes)
  * feature_variance_probe   (the router signal: is the object multi-material?)

STEP 1 (new) — FOUNDATION FEATURES (diagram box 2). DINOv3 stops being a
mask-maker and becomes a per-atom feature oracle:
  * feature_boundary_map     B_dino: an INDEPENDENT boundary signal (not derived
                             from the support threshold).
  * object_bg_prototypes     the (object, background) prototypes for a detection,
                             reused so per-atom objectness is consistent with S.
  * region_feature / feature_objectness / region_coherence / same_material
                             the per-atom DINO facts that let ASP reason over
                             SEMANTICS as well as geometry.
  * mask_edge_on_bdino       QA: does the mask boundary sit on a real feature
                             ridge, or did the mask balloon into flat seabed?

Nothing here is class-specific. Region/contour atom logic lives in M2/M3.

Deps: torch, numpy, PIL, scipy. (cv2 only needed downstream, not here.)
"""
import os
import re
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 16


# ============================================================ checkpoint loading
def _looks_like_bare_vit(sd):
    return any(k == "cls_token" or k.startswith("blocks.") for k in sd)


def extract_dinov3_state(raw, prefer_ema=True, verbose=True):
    sd = raw
    if isinstance(sd, dict) and not _looks_like_bare_vit(sd):
        if verbose:
            print(f"[ckpt] top-level keys: {list(sd.keys())}")
        cands = []
        if prefer_ema and "ema" in sd and isinstance(sd["ema"], dict):
            cands.append(sd["ema"].get("module", sd["ema"]))
        if "model" in sd and isinstance(sd["model"], dict):
            cands.append(sd["model"])
        cands.append(sd)
        for cand in cands:
            if not isinstance(cand, dict):
                continue
            if _looks_like_bare_vit(cand):
                return cand
            prefixes = set()
            for k in cand.keys():
                m = re.search(r"(.*?dinov3\.)", k)
                if m:
                    prefixes.add(m.group(1))
            for pref in sorted(prefixes, key=len):
                sub = {k[len(pref):]: v for k, v in cand.items() if k.startswith(pref)}
                if _looks_like_bare_vit(sub):
                    if verbose:
                        print(f"[ckpt] extracted DINOv3 from prefix '{pref}' ({len(sub)} tensors)")
                    return sub
    return sd


def load_dinov3(repo_root, weights, weights_kind="deim", prefer_ema=True, device="cuda"):
    bdir = os.path.join(repo_root, "engine", "backbone")
    if bdir not in sys.path:
        sys.path.insert(0, bdir)
    from dinov3 import DinoVisionTransformer
    raw = torch.load(weights, map_location="cpu", weights_only=False)
    if weights_kind == "pretrained":
        sd = raw if _looks_like_bare_vit(raw) else extract_dinov3_state(raw, prefer_ema)
    elif weights_kind == "deim":
        sd = extract_dinov3_state(raw, prefer_ema)
    else:
        sd = raw if (isinstance(raw, dict) and _looks_like_bare_vit(raw)) \
            else extract_dinov3_state(raw, prefer_ema)
    model = DinoVisionTransformer(name="dinov3_vits16")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded {len(sd) - len(unexpected)} tensors "
          f"(kind={weights_kind}, ema={prefer_ema})")
    if len(sd) - len(unexpected) < 50:
        print("[ERROR] very few tensors loaded - extraction probably failed.")
    model.eval().to(device)
    return model


# ============================================================ crop + model input
def crop_with_margin(image_np, box, margin=0.15):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin, h * margin
    H, W = image_np.shape[:2]
    x1m = int(max(0, np.floor(x1 - mx)))
    y1m = int(max(0, np.floor(y1 - my)))
    x2m = int(min(W, np.ceil(x2 + mx)))
    y2m = int(min(H, np.ceil(y2 + my)))
    return image_np[y1m:y2m, x1m:x2m], (x1m, y1m, x2m, y2m)


def to_model_input(crop_np, min_grid, max_side, device):
    h, w = crop_np.shape[:2]
    short, long = min(h, w), max(h, w)
    scale = max(1.0, (min_grid * PATCH) / short)
    if long * scale > max_side:
        scale = max_side / long
    rh = max(PATCH, int(round(h * scale / PATCH)) * PATCH)
    rw = max(PATCH, int(round(w * scale / PATCH)) * PATCH)
    img = Image.fromarray(crop_np).resize((rw, rh), Image.BICUBIC)
    arr = (np.asarray(img).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device), (rh, rw)


@torch.no_grad()
def extract_tokens(model, ten, layer):
    fmap = model.get_intermediate_layers(ten, n=[layer], reshape=True, norm=True)[0]
    _, C, gh, gw = fmap.shape
    tok = F.normalize(fmap[0].permute(1, 2, 0).reshape(gh * gw, C), dim=1)
    return tok, (gh, gw)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def _as_np(tok):
    return tok.detach().float().cpu().numpy() if hasattr(tok, "detach") else np.asarray(tok)


# ============================================================ grid geometry / prototypes
# Extracted out of compute_support so the SAME object/background prototypes can be
# reused for per-atom feature_objectness (Step 1). The math below is bit-for-bit
# what compute_support used to do inline.
def _grid_centers(grid, model_hw, crop_hw):
    """Crop-pixel (x, y) centre of every token patch (flat, row-major)."""
    gh, gw = grid
    rh, rw = model_hw
    ch, cw = crop_hw
    ys = (np.arange(gh) + 0.5) * PATCH * (ch / rh)
    xs = (np.arange(gw) + 0.5) * PATCH * (cw / rw)
    return np.tile(xs, gh), np.repeat(ys, gw)


def _inside_ring_seed(grid, model_hw, inner_box, crop_hw, seed_frac):
    cx, cy = _grid_centers(grid, model_hw, crop_hw)
    bx1, by1, bx2, by2 = inner_box
    inside = (cx >= bx1) & (cx <= bx2) & (cy >= by1) & (cy <= by2)
    if inside.sum() < 1:
        inside = np.ones(grid[0] * grid[1], bool)
    ring = ~inside
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    hwx, hwy = (bx2 - bx1) * seed_frac / 2, (by2 - by1) * seed_frac / 2
    seed = (cx >= bcx - hwx) & (cx <= bcx + hwx) & (cy >= bcy - hwy) & (cy <= bcy + hwy)
    if seed.sum() < 1:
        seed = inside
    return inside, ring, seed


def _refine_obj_proto(t, inside, seed):
    """Center-seed prototype + one percentile-50 refinement (== compute_support)."""
    obj = _unit(t[seed].mean(0))
    sim = t @ obj
    if inside.sum() >= 4:
        thr = np.percentile(sim[inside], 50)
        refine = inside & (sim >= thr)
        if refine.sum() >= 2:
            obj = _unit(t[refine].mean(0))
            sim = t @ obj
    return obj, sim


def object_bg_prototypes(tok, grid, model_hw, inner_box, crop_hw, seed_frac=0.5):
    """The (object, background) unit prototypes for THIS detection, refined exactly
    as compute_support does. Reuse these for every atom's feature_objectness so the
    per-atom semantics are consistent with the support map. bg may be None."""
    t = _as_np(tok)
    inside, ring, seed = _inside_ring_seed(grid, model_hw, inner_box, crop_hw, seed_frac)
    obj, _ = _refine_obj_proto(t, inside, seed)
    bg = _unit(t[ring].mean(0)) if ring.sum() >= 1 else None
    return obj, bg


# ============================================================ support map (v2)
def compute_support(tok, grid, model_hw, inner_box, crop_hw, seed_frac=0.5):
    """Center-seeded object prototype vs background ring, dynamic-range gated.
    Returns (support_grid HxW in 0-100, gate in 0-1).

    (Refactored to share _inside_ring_seed / _refine_obj_proto with
    object_bg_prototypes; numerically identical to the previous version.)"""
    gh, gw = grid
    t = _as_np(tok)
    inside, ring, seed = _inside_ring_seed(grid, model_hw, inner_box, crop_hw, seed_frac)
    obj, sim = _refine_obj_proto(t, inside, seed)
    if ring.sum() >= 1:
        contrast = np.clip(sim - (t @ _unit(t[ring].mean(0))), 0, None)
    else:
        contrast = np.clip(sim, 0, None)
    rng = np.percentile(sim, 95) - np.percentile(sim, 5)
    gate = float(np.clip(rng / 0.15, 0, 1))

    def mm(v):
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        return np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)

    support = (0.5 * mm(sim) + 0.5 * mm(contrast)) * 100 * gate
    return support.reshape(gh, gw), gate


# ============================================================ cleaned mask
def largest_cc(mask):
    lbl, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def clean_mask(support_grid, sup_thr_frac=0.55):
    """Grid-resolution boolean object mask: threshold at frac*max -> keep the
    largest connected component -> fill interior holes. This is THE mask every
    downstream stage uses. Returns None if nothing survives."""
    smax = support_grid.max() + 1e-6
    m = support_grid >= sup_thr_frac * smax
    if m.sum() < 2:
        return None
    m = largest_cc(m)
    m = ndi.binary_fill_holes(m)
    return m


def upsample_mask(mask_grid, crop_hw):
    """Nearest-upsample a grid-res boolean mask to crop pixel resolution."""
    ch, cw = crop_hw
    m = Image.fromarray((mask_grid.astype(np.uint8) * 255))
    m = m.resize((cw, ch), Image.NEAREST)
    return np.asarray(m) > 0


def upsample_grid(grid_map, crop_hw, mode="bilinear"):
    """Upsample a gh x gw float map (support, B_dino, ...) to crop pixel
    resolution. Bilinear by default; pass mode='nearest' for hard maps."""
    ch, cw = crop_hw
    t = torch.from_numpy(np.ascontiguousarray(grid_map.astype(np.float32)))[None, None]
    kw = dict(align_corners=False) if mode != "nearest" else {}
    return F.interpolate(t, size=(ch, cw), mode=mode, **kw)[0, 0].numpy()


def mask_quality(support_grid, mask_grid):
    """Per-detection mask measurements from the support map (no model calls).

    Emits RAW features, not a verdict — ASP decides what 'low quality' means.
    The key one is BOUNDARY SHARPNESS: support just inside the mask edge minus
    support just outside it. A real object edge is a steep support drop (big
    sharpness); a ballooned mask sits on a gentle support slope out in the
    seabed, so the drop across its edge is small. Unlike region contrast (which
    saturates on visually-distinct urchins), this is local to the boundary and
    stays informative.

    Returns: s_in, s_out, contrast, mask_frac, inner_edge, outer_edge,
             boundary_sharpness  (all on the 0-100 support scale / 0-1 for frac).
    None-safe.
    """
    if mask_grid is None:
        return dict(s_in=0.0, s_out=0.0, contrast=0.0, mask_frac=0.0,
                    inner_edge=0.0, outer_edge=0.0, boundary_sharpness=0.0)
    inside = mask_grid
    outside = ~mask_grid
    s_in = float(support_grid[inside].mean()) if inside.any() else 0.0
    s_out = float(support_grid[outside].mean()) if outside.any() else 0.0
    er = ndi.binary_erosion(mask_grid, iterations=1)
    dil = ndi.binary_dilation(mask_grid, iterations=1)
    inner_ring = mask_grid & ~er
    outer_ring = dil & ~mask_grid
    inner_edge = float(support_grid[inner_ring].mean()) if inner_ring.any() else s_in
    outer_edge = float(support_grid[outer_ring].mean()) if outer_ring.any() else 0.0
    return dict(s_in=s_in, s_out=s_out, contrast=s_in - s_out,
                mask_frac=float(inside.mean()),
                inner_edge=inner_edge, outer_edge=outer_edge,
                boundary_sharpness=inner_edge - outer_edge)


# ============================================================ STEP 1: foundation features
# B_dino + per-atom DINOv3 feature facts. These turn DINOv3 from a mask-maker into
# a per-atom feature oracle: every atom can carry an objectness / coherence /
# same-material fact, so the downstream reasoner sees SEMANTICS as well as
# geometry. Nothing here is class-specific.

def feature_boundary_map(tok, grid, connectivity=8):
    """B_dino. Per-patch boundary energy = MAX cosine-distance to grid neighbours.
    Tokens are already L2-normed (extract_tokens), so cosine sim = dot and
    distance = 1 - dot. A patch straddling the object edge differs sharply from a
    neighbour -> a high-energy ridge ON the boundary; a uniform body or uniform
    seabed -> ~0.

    Crucially this is an INDEPENDENT boundary signal: unlike the mask edge /
    silhouette / mask-refereed Canny, it is NOT derived from the support
    threshold. Where it disagrees with the mask edge, the MASK is the suspect
    (ballooned into a flat feature field).

    Returns a gh x gw float map (cosine distance, typically 0..~0.6). NOT
    normalized — the caller scales for display / fact emission."""
    gh, gw = grid
    Fm = _as_np(tok).reshape(gh, gw, -1)
    bmap = np.zeros((gh, gw), np.float32)

    def acc(a_sl, b_sl):
        d = (1.0 - (Fm[a_sl] * Fm[b_sl]).sum(-1)).astype(np.float32)
        bmap[a_sl] = np.maximum(bmap[a_sl], d)
        bmap[b_sl] = np.maximum(bmap[b_sl], d)

    # 4-connectivity: right, down
    acc((slice(None), slice(0, gw - 1)), (slice(None), slice(1, gw)))
    acc((slice(0, gh - 1), slice(None)), (slice(1, gh), slice(None)))
    if connectivity == 8:
        # main diagonal (i,j)-(i+1,j+1)
        acc((slice(0, gh - 1), slice(0, gw - 1)), (slice(1, gh), slice(1, gw)))
        # anti diagonal (i,j+1)-(i+1,j)
        acc((slice(0, gh - 1), slice(1, gw)), (slice(1, gh), slice(0, gw - 1)))
    return bmap


def region_feature(tok, grid, mask_grid):
    """Mean L2-normed DINOv3 feature over the tokens inside a grid-res boolean
    mask. Returns a unit vector, or None if the mask is empty / None."""
    if mask_grid is None:
        return None
    t = _as_np(tok)
    idx = mask_grid.reshape(-1).astype(bool)
    if idx.sum() < 1:
        return None
    return _unit(t[idx].mean(0))


def feature_objectness(region_feat, obj_proto, bg_proto, calibrated=True):
    """Object-minus-background cosine margin of a region's mean feature, 0-100.
    High = semantically object-like; ~0 = looks like the seabed ring.

    calibrated=True (default) divides the margin by the detection's OWN maximum
    separable margin, 1 - cos(obj_proto, bg_proto). DINOv3 patch means aren't
    orthogonal, so a clean body's raw margin is only ~0.2 (reads as 20 and looks
    like failure); normalizing by what THIS detection could maximally separate
    makes a clean body read ~80-100 while a region drifting onto seabed still
    collapses toward 0. No magic constant — the scale is the data's own
    separability. Per-atom discrimination is unchanged; only the scale is fixed.
    calibrated=False returns the raw margin*100 (the old behaviour)."""
    if region_feat is None or obj_proto is None:
        return 0.0
    so = float(region_feat @ obj_proto)
    sb = float(region_feat @ bg_proto) if bg_proto is not None else 0.0
    margin = so - sb
    if calibrated and bg_proto is not None:
        denom = 1.0 - float(obj_proto @ bg_proto)
        return float(np.clip(margin / (denom + 1e-6), 0.0, 1.0) * 100)
    return float(np.clip(margin, 0.0, 1.0) * 100)


def region_coherence(tok, grid, mask_grid):
    """Mean cosine of inside tokens to their own centroid, 0-100. High = one
    uniform material/part (a clean atom); low = the region straddles two things
    (a junk atom). This is 100 * (1 - dispersion) from feature_variance_probe."""
    if mask_grid is None:
        return 0.0
    t = _as_np(tok)
    idx = mask_grid.reshape(-1).astype(bool)
    inside = t[idx]
    if len(inside) < 2:
        return 0.0
    centroid = _unit(inside.mean(0))
    return float(np.clip((inside @ centroid).mean(), 0.0, 1.0) * 100)


def same_material(feat_a, feat_b):
    """Cosine between two region mean features, 0-100. High = same surface (all
    spikes of one animal; body and its own arms); low = a foreign object (a shell
    sitting on the urchin). RAW — ASP groups/splits on it."""
    if feat_a is None or feat_b is None:
        return 0.0
    return float(np.clip(float(feat_a @ feat_b), 0.0, 1.0) * 100)


def mask_edge_on_bdino(bdino_grid, mask_grid):
    """Does the mask boundary sit on a real DINOv3 feature ridge? Compares B_dino
    ON the mask perimeter vs the mask INTERIOR (grid resolution).

      perim_mean    : mean B_dino over the 1-token boundary ring of the mask
      interior_mean : mean B_dino strictly inside
      ratio         : perim_mean / (interior_mean + eps)

    ratio >> 1 -> the mask edge lands on a feature discontinuity (trustworthy).
    ratio ~ 1  -> the edge sits in a flat feature field (mask likely ballooned
                  into uniform seabed). None-safe."""
    if mask_grid is None:
        return dict(perim_mean=0.0, interior_mean=0.0, ratio=0.0)
    er = ndi.binary_erosion(mask_grid, iterations=1)
    perim = mask_grid & ~er
    interior = er
    pm = float(bdino_grid[perim].mean()) if perim.any() else 0.0
    im = float(bdino_grid[interior].mean()) if interior.any() else 0.0
    return dict(perim_mean=pm, interior_mean=im, ratio=float(pm / (im + 1e-6)))


# ============================================================ router probe
def _two_means_separation(X, iters=12, seed=0):
    """Tiny k=2 on L2-normed rows. Returns separation = ||c1-c2|| / mean spread.
    High => the inside tokens fall into two distinct feature groups
    (multi-material object); low => one uniform surface."""
    if len(X) < 6:
        return 0.0
    rng = np.random.default_rng(seed)
    # init on the top principal axis (most-separated pair of projections)
    Xc = X - X.mean(0)
    try:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt[0]
        c = np.stack([X[proj.argmin()], X[proj.argmax()]])
    except np.linalg.LinAlgError:
        c = X[rng.choice(len(X), 2, replace=False)]
    for _ in range(iters):
        d = np.stack([((X - c[k]) ** 2).sum(1) for k in range(2)], 1)
        lab = d.argmin(1)
        if lab.min() == lab.max():
            return 0.0
        newc = np.stack([X[lab == k].mean(0) for k in range(2)])
        if np.allclose(newc, c):
            c = newc
            break
        c = newc
    between = float(np.linalg.norm(c[0] - c[1]))
    within = float(np.mean([np.linalg.norm(X[lab == k] - c[k], axis=1).mean()
                            for k in range(2)]) + 1e-8)
    return between / within


def _spherical_kmeans(X, k, iters=15, seed=0):
    """k-means on L2-normalized vectors (cosine). Farthest-first init."""
    n = len(X)
    if n <= k:
        return np.arange(n), X.copy()
    rng = np.random.default_rng(seed)
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        sims = np.max(X @ np.stack(centers).T, axis=1)
        centers.append(X[int(np.argmin(sims))])           # farthest from current set
    C = np.stack(centers)
    lab = np.zeros(n, int)
    for _ in range(iters):
        lab = (X @ C.T).argmax(1)
        newC = np.stack([_unit(X[lab == j].mean(0)) if (lab == j).any() else C[j]
                         for j in range(len(C))])
        if np.allclose(newC, C):
            break
        C = newC
    return lab, C


def _merge_similar(lab, C, merge_cos):
    """Union clusters whose centers have cosine sim > merge_cos; relabel.
    Auto-selects effective k: a uniform object collapses to 1 cluster."""
    k = len(C)
    parent = list(range(k))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    sim = C @ C.T
    for i in range(k):
        for j in range(i + 1, k):
            if sim[i, j] > merge_cos:
                parent[find(i)] = find(j)
    roots = {}
    for old in range(k):
        roots.setdefault(find(old), len(roots))
    return np.array([roots[find(l)] for l in lab])


def cluster_inside_tokens(tok, grid, mask_grid, max_k=4, merge_cos=0.92):
    """Cluster the DINOv3 tokens INSIDE the mask by embedding similarity, so a
    foreign surface on the object (a shell on an urchin) separates from the body
    by FEATURE, not geometry. Returns None if too few tokens, else dict with the
    inside-token features X, flat grid indices, per-token labels, and body_label
    (largest cluster). Auto-selects effective k via merge_cos."""
    t = _as_np(tok)
    inside_idx = np.where(mask_grid.reshape(-1).astype(bool))[0]
    if len(inside_idx) < 12:
        return None
    X = t[inside_idx]
    lab, C = _spherical_kmeans(X, k=min(max_k, max(2, len(inside_idx) // 8)))
    lab = _merge_similar(lab, C, merge_cos)
    counts = np.bincount(lab)
    return dict(X=X, inside_idx=inside_idx, labels=lab, body_label=int(counts.argmax()),
                n_clusters=int(counts.size))


def feature_variance_probe(tok, grid, mask_grid):
    """Router signal. Are the DINOv3 features INSIDE the object structured
    (multi-material -> feature clustering can recover parts) or flat (uniform
    surface -> only geometry can split it)?

    Returns dict:
      n_inside    : #tokens inside the mask
      dispersion  : mean cosine distance of inside tokens to their centroid
      bimodality  : 2-means between/within separation (the routing number)

    Heuristic used by M2: bimodality high  -> feature branch (object_element atoms)
                          bimodality low   -> geometry branch (geometric_form only)
    """
    t = _as_np(tok)
    idx = mask_grid.reshape(-1).astype(bool)
    inside = t[idx]
    if len(inside) < 4:
        return dict(n_inside=int(len(inside)), dispersion=0.0, bimodality=0.0)
    centroid = _unit(inside.mean(0))
    cos = inside @ centroid
    dispersion = float(1.0 - cos.mean())
    bimodality = _two_means_separation(inside)
    return dict(n_inside=int(len(inside)), dispersion=dispersion, bimodality=bimodality)


# ============================================================ I/O helpers
def load_predictions(pred_json, box_format="xywh"):
    import json
    from collections import defaultdict
    with open(pred_json) as f:
        preds = json.load(f)
    if isinstance(preds, dict) and "annotations" in preds:
        preds = preds["annotations"]
    by = defaultdict(list)
    for p in preds:
        x, y, a, b = p["bbox"]
        box = [x, y, x + a, y + b] if box_format == "xywh" else [x, y, a, b]
        by[p["image_id"]].append({"box": box, "score": float(p.get("score", 1.0)),
                                  "cat": p["category_id"]})
    return by


def load_gt_index(gt_json):
    import json
    with open(gt_json) as f:
        gt = json.load(f)
    id2file = {im["id"]: im["file_name"] for im in gt["images"]}
    name2cat = {c["name"]: c["id"] for c in gt["categories"]}
    cat2name = {c["id"]: c["name"] for c in gt["categories"]}
    # optional: per-image GT polygons by category, for QA against segmentation
    from collections import defaultdict
    gt_seg = defaultdict(list)
    for a in gt.get("annotations", []):
        if a.get("segmentation"):
            x, y, w, h = a["bbox"]
            gt_seg[a["image_id"]].append(
                dict(cat=a["category_id"], box=[x, y, x + w, y + h], seg=a["segmentation"]))
    return id2file, name2cat, cat2name, gt_seg