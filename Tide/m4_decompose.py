#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m4_decompose.py  —  Milestone 4b: the urchin DECOMPOSITION (data-grounded).

The M4 diagnostic falsified PiDiNet-for-spines (PiDiNet active 3.2% in the
annulus) and PROVED the spine signal is a radial B_dino HALO (B_dino active
47.7%, halo extent x1.33, coverage 87%). So this version:

  * BODY   = M2 ellipse (shape) + silhouette boundary (M3 QA winner, 4px).
  * SHELLS = bright foreign surfaces INSIDE the ellipse, from PiDiNet + intensity
             (PiDiNet's one real job here). DINOv3 bimodality corroborates.
  * SPINES = per-sector spine_halo atoms read from the B_dino radial halo
             (primary), corroborated by SUPPORT falloff (secondary). NO line
             fitting -- the diagnostic showed spines are a diffuse radial FIELD,
             not separable segments, so we describe the field per angular sector:
             radial extent, halo strength, support falloff -> confidence.

SHELL OCCLUDES SPINES: a sector overlapping a shell's arc emits no spine_halo
atom (the shell, not spines, occupies that arc).

Detection rollups (for ASP): halo_coverage, mean_spine_extent, mean_halo_strength,
radial_symmetry, spine_sector_count, occluded_sectors, bimodality.
PER-ATOM confidence(id,0-100) on every atom.

Run on your machine. Deps: torch, numpy, cv2, PIL, scipy, matplotlib, core.py,
m2_regions.py, m3_contours.py, pidinet_edge.py + pidinet_pkg/ + table7_pidinet.pth.
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
from matplotlib import cm

import core
from m2_regions import (ellipse_body_atom, fit_ellipse_to_mask, ellipse_to_mask,
                        iou_u8, shape_descriptors)
from m3_contours import silhouette_contours
from pidinet_edge import PiDiNetEdger


# ============================================================ geometry helpers
def _angdist_deg(a, b, period=360.0):
    d = abs((a - b) % period)
    return min(d, period - d)


def ellipse_radius_at(ellipse, theta_deg):
    """Centre-to-boundary distance along image-frame angle theta (deg)."""
    _, _, d1, d2, ang = ellipse
    a, b = d1 / 2.0, d2 / 2.0
    psi = math.radians(theta_deg) - math.radians(ang)
    ca, sa = math.cos(psi), math.sin(psi)
    return (a * b) / (math.sqrt((b * ca) ** 2 + (a * sa) ** 2) + 1e-9)


def _pixel_to_grid(reg_bool, grid):
    gh, gw = grid
    r = Image.fromarray((reg_bool.astype(np.uint8) * 255)).resize((gw, gh), Image.NEAREST)
    return np.asarray(r) > 0


def annulus_zone(ellipse, crop_hw, det_box, annulus_scale):
    ch, cw = crop_hw
    inner = ellipse_to_mask((ch, cw), ellipse, 1.0) > 0
    outer = ellipse_to_mask((ch, cw), ellipse, annulus_scale) > 0
    box = np.zeros((ch, cw), bool)
    bx1, by1, bx2, by2 = [int(round(v)) for v in det_box]
    box[max(0, by1):min(ch, by2), max(0, bx1):min(cw, bx2)] = True
    return inner, (outer & ~inner & box)


def _safe_mean(v):
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


# ============================================================ shell (internal_region) atoms
def shell_atoms(crop_rgb, ellipse, inner_mask, prob, tok, grid, support_up, body_feat, p):
    ch, cw = crop_rgb.shape[:2]
    cx, cy = ellipse[0], ellipse[1]
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    inside_vals = gray[inner_mask]
    if inside_vals.size < 20:
        return []
    mu, sd = float(inside_vals.mean()), float(inside_vals.std()) + 1e-6
    bright = ((gray > mu + p["shell_bright_k"] * sd) & inner_mask).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl = cv2.connectedComponents(bright)
    body_area = float(inner_mask.sum()) + 1e-6
    atoms = []
    for i in range(1, n):
        reg = lbl == i
        area = int(reg.sum())
        if area < p["shell_min_frac"] * body_area:
            continue
        pidi_here = float(prob[reg].mean())
        if pidi_here < p["shell_min_pidi"]:
            continue
        ys, xs = np.where(reg)
        rcx, rcy = float(xs.mean()), float(ys.mean())
        sfeat = core.region_feature(tok, grid, _pixel_to_grid(reg, grid))
        sm = core.same_material(sfeat, body_feat) if sfeat is not None else 100.0
        contrast = 100.0 - sm
        sup = float(support_up[reg].mean())
        conf = int(round(0.5 * contrast + 0.3 * pidi_here * 100 + 0.2 * sup))
        ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) % 360.0)
        atoms.append(dict(id=None, role="internal_region", atype="shell_candidate",
                          kind="region", centroid=(rcx, rcy),
                          bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                          mask=(reg.astype(np.uint8) * 255), area=area,
                          contrast=contrast, pidi=pidi_here * 100, support=sup,
                          confidence=conf, angles=ang))
    atoms.sort(key=lambda a: -a["confidence"])
    return atoms[:p["max_shells"]]


def _sector_blocked_by_shell(lo, hi, shells, margin_deg=6.0):
    """True if angular sector [lo,hi) overlaps any shell's arc (+margin)."""
    for s in shells:
        amin, amax = float(s["angles"].min()) - margin_deg, float(s["angles"].max()) + margin_deg
        # overlap test on the circle (treat shell arc as [amin,amax], no wrap for simplicity)
        if not (hi <= amin or lo >= amax):
            return True
    return False


# ============================================================ spine_halo (per-sector) atoms
def halo_sector_atoms(bdino_up, support_up, ellipse, crop_hw, shells, p):
    """Read the B_dino radial halo into one spine_halo atom per angular sector.
    For each sector: average B_dino & support along rays from the body edge
    outward; measure halo EXTENT (how far B_dino stays elevated above the far
    seabed), halo STRENGTH (edge-vs-far B_dino contrast), and SUPPORT FALLOFF
    (corroboration). A sector emits an atom only if a real halo is present
    (extent>1.1 and strength>=min_contrast) and it is not shell-occluded.

    Returns (atoms, rollup)."""
    ch, cw = crop_hw
    cx, cy = ellipse[0], ellipse[1]
    N = p["n_sectors"]
    rays = p["rays_per_sector"]
    r_steps = p["r_steps"]
    rr_grid = np.linspace(1.0, p["annulus_scale"], r_steps)
    near = rr_grid <= 1.15
    far = rr_grid >= max(1.75, p["annulus_scale"] - 0.05)

    atoms, extents, strengths, present, occluded = [], [], [], 0, 0
    sec_w = 360.0 / N
    for s in range(N):
        lo, hi = s * sec_w, (s + 1) * sec_w
        mid = (lo + hi) / 2.0
        if _sector_blocked_by_shell(lo, hi, shells):
            occluded += 1
            continue
        # sample rays across the sector
        bd_rows, sp_rows = [], []
        for a in np.linspace(lo, hi, rays, endpoint=False) + sec_w / (2 * rays):
            br = ellipse_radius_at(ellipse, a)
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            bd_r = np.full(r_steps, np.nan, np.float32)
            sp_r = np.full(r_steps, np.nan, np.float32)
            for ri, rr in enumerate(rr_grid):
                x, y = int(round(cx + br * rr * ca)), int(round(cy + br * rr * sa))
                if 0 <= y < ch and 0 <= x < cw:
                    bd_r[ri] = bdino_up[y, x]; sp_r[ri] = support_up[y, x]
            bd_rows.append(bd_r); sp_rows.append(sp_r)
        bd = np.nanmean(np.stack(bd_rows), axis=0)
        sp = np.nanmean(np.stack(sp_rows), axis=0)
        base = _safe_mean(bd[near]); bg = _safe_mean(bd[far])
        if not math.isfinite(base):
            continue
        if not math.isfinite(bg):
            bg = 0.0
        strength = max(0.0, base - bg)
        ext = 1.0
        if strength >= p["min_contrast"]:
            thr = bg + 0.5 * strength
            for ri in range(r_steps):
                if math.isfinite(bd[ri]) and bd[ri] >= thr:
                    ext = rr_grid[ri]
                else:
                    break
        if ext <= 1.1 or strength < p["min_contrast"]:
            continue                                   # bare edge / seabed -> no spine here
        sup_fall = max(0.0, _safe_mean(sp[near]) - (_safe_mean(sp[far]) if math.isfinite(_safe_mean(sp[far])) else 0.0))

        cb = np.clip(strength / p["strength_ref"], 0, 1) * 100
        ce = np.clip((ext - 1.0) / (p["extent_ref"] - 1.0), 0, 1) * 100
        cs = np.clip(sup_fall / p["falloff_ref"], 0, 1) * 100
        conf = int(round(0.5 * cb + 0.3 * ce + 0.2 * cs))

        br_mid = ellipse_radius_at(ellipse, mid)
        rmid = br_mid * (1.0 + ext) / 2.0
        hx = cx + rmid * math.cos(math.radians(mid))
        hy = cy + rmid * math.sin(math.radians(mid))
        present += 1; extents.append(ext); strengths.append(strength)
        atoms.append(dict(id=f"h{s+1}", role="spine_halo", kind="halo",
                          sector=(int(round(lo)), int(round(hi))), direction=int(round(mid)),
                          extent=ext, strength=strength, support_falloff=sup_fall,
                          confidence=conf, centroid=(hx, hy),
                          bbox=(int(hx-3), int(hy-3), int(hx+3), int(hy+3)),
                          inner_pt=(cx + br_mid*math.cos(math.radians(mid)),
                                    cy + br_mid*math.sin(math.radians(mid))),
                          outer_pt=(cx + br_mid*ext*math.cos(math.radians(mid)),
                                    cy + br_mid*ext*math.sin(math.radians(mid)))))

    cover = present / float(N)
    mean_ext = float(np.mean(extents)) if extents else 1.0
    mean_str = float(np.mean(strengths)) if strengths else 0.0
    sym = (1.0 - min(1.0, float(np.std(extents)) / (float(np.mean(extents)) + 1e-6))) \
        if len(extents) >= 2 else (1.0 if extents else 0.0)
    roll = dict(coverage=cover, mean_extent=mean_ext, mean_strength=mean_str,
                symmetry=sym, present=present, occluded=occluded, n_sectors=N)
    return atoms, roll


# ============================================================ facts
def body_confidence(body):
    return int(round(0.4 * body["objness"] + 0.3 * (100 - body["fit_error"] * 100)
                     + 0.3 * body["support"]))


def write_facts(path, det_box_crop, body, silh, shells, halos, roll, probe, mq,
                oid="o1", cls="object"):
    L = ["% M4b decomposition: ellipse body + silhouette + shells + per-sector spine_halo",
         "% spines are a radial B_dino halo (diagnostic-proven), described per sector.",
         "% every atom carries confidence(atom, 0-100).", ""]
    bx = [int(round(v)) for v in det_box_crop]
    L += [f"detection({oid}).", f"class({oid},{cls}).",
          f"proposal_box({oid},box({bx[0]},{bx[1]},{bx[2]},{bx[3]})).",
          f"bimodality({oid},{int(round(probe.get('bimodality', 0.0) * 100))})."]
    if mq is not None:
        L += [f"mask_contrast({oid},{int(round(mq['contrast']))}).",
              f"boundary_sharpness({oid},{int(round(mq['boundary_sharpness']))})."]
    # spine-halo detection rollups
    L += [f"halo_coverage({oid},{int(round(roll['coverage']*100))}).",
          f"mean_spine_extent({oid},{int(round(roll['mean_extent']*100))}).",
          f"mean_halo_strength({oid},{int(round(roll['mean_strength']*100))}).",
          f"radial_symmetry({oid},{int(round(roll['symmetry']*100))}).",
          f"spine_sector_count({oid},{roll['present']}).",
          f"occluded_sectors({oid},{roll['occluded']}).",
          f"sufficient_spine_coverage({oid}) :- halo_coverage({oid},C), C >= 40."]

    # body
    bcx, bcy = [int(round(v)) for v in body["centroid"]]
    ecx, ecy, ed1, ed2, eang = body["ellipse"]
    bb = [int(round(v)) for v in body["bbox"]]
    L += ["", f"atom(body1).", f"belongs_to_detection(body1,{oid}).",
          f"atom_kind(body1,geometric_form).", f"atom_type(body1,body_form).",
          f"shape_class(body1,ellipse).",
          f"ellipse(body1,point({int(round(ecx))},{int(round(ecy))}),"
          f"axes({int(round(ed1))},{int(round(ed2))}),angle({int(round(eang))})).",
          f"centroid(body1,point({bcx},{bcy})).",
          f"atom_box(body1,box({bb[0]},{bb[1]},{bb[2]},{bb[3]})).",
          f"fit_error(body1,{int(round(body['fit_error']*100))}).",
          f"feature_objectness(body1,{int(round(body['objness']))}).",
          f"region_coherence(body1,{int(round(body['coher']))}).",
          f"support(body1,{int(round(body['support']))}).",
          f"confidence(body1,{body_confidence(body)}).", f"inside(body1,{oid})."]

    # silhouette boundary
    if silh is not None:
        scx, scy = [int(round(v)) for v in silh["centroid"]]
        sb = [int(round(v)) for v in silh["bbox"]]
        L += ["", f"atom(sil1).", f"belongs_to_detection(sil1,{oid}).",
              f"atom_kind(sil1,contour).", f"atom_type(sil1,outer_contour_fragment).",
              f"contour_source(sil1,silhouette).", f"length(sil1,{silh['length']}).",
              f"centroid(sil1,point({scx},{scy})).",
              f"atom_box(sil1,box({sb[0]},{sb[1]},{sb[2]},{sb[3]})).",
              f"support(sil1,{int(round(silh['support']))}).",
              f"confidence(sil1,{int(round(silh['support']))}).", f"inside(sil1,{oid}).",
              f"supports_contour(body1,sil1)."]

    # shells
    for k, s in enumerate(shells, 1):
        sid = f"sh{k}"; s["id"] = sid
        cx, cy = [int(round(v)) for v in s["centroid"]]
        bb = [int(round(v)) for v in s["bbox"]]
        amin, amax = int(s["angles"].min()), int(s["angles"].max())
        L += ["", f"atom({sid}).", f"belongs_to_detection({sid},{oid}).",
              f"atom_kind({sid},region).", f"atom_type({sid},{s['atype']}).",
              f"centroid({sid},point({cx},{cy})).",
              f"atom_box({sid},box({bb[0]},{bb[1]},{bb[2]},{bb[3]})).",
              f"area({sid},{s['area']}).",
              f"feature_contrast({sid},{int(round(s['contrast']))}).",
              f"edge_strength({sid},{int(round(s['pidi']))}).",
              f"support({sid},{int(round(s['support']))}).",
              f"confidence({sid},{s['confidence']}).",
              f"inside({sid},{oid}).", f"on_body({sid},body1).",
              f"occludes_arc({sid},body1,{amin},{amax})."]

    # spine_halo sectors
    for h in halos:
        hid = h["id"]
        cx, cy = [int(round(v)) for v in h["centroid"]]
        bb = [int(round(v)) for v in h["bbox"]]
        lo, hi = h["sector"]
        L += ["", f"atom({hid}).", f"belongs_to_detection({hid},{oid}).",
              f"atom_kind({hid},halo).", f"atom_type({hid},spine_halo).",
              f"sector({hid},{lo},{hi}).", f"direction({hid},{h['direction']}).",
              f"centroid({hid},point({cx},{cy})).",
              f"atom_box({hid},box({bb[0]},{bb[1]},{bb[2]},{bb[3]})).",
              f"halo_extent({hid},{int(round(h['extent']*100))}).",
              f"halo_strength({hid},{int(round(h['strength']*100))}).",
              f"support_falloff({hid},{int(round(h['support_falloff']))}).",
              f"confidence({hid},{h['confidence']}).", f"inside({hid},{oid}).",
              f"radial_to({hid},body1).", f"points_outward({hid},body1)."]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ============================================================ visualize
def visualize(crop_rgb, det_box_crop, body, silh, shells, halos, roll, annulus,
              out_path, title):
    H, W = crop_rgb.shape[:2]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    bx1, by1, bx2, by2 = det_box_crop

    ax[0].imshow(crop_rgb)
    ax[0].add_patch(Rectangle((bx1, by1), bx2-bx1, by2-by1, fill=False, ec="lime", lw=2))
    ov = np.zeros((H, W, 4)); ov[annulus] = [0, 0.6, 1, 0.22]; ax[0].imshow(ov)
    ax[0].contour(body["mask"].astype(float), levels=[127], colors="yellow", linewidths=1.5)
    ax[0].set_title("ellipse body (yellow) + spine annulus (blue)")

    ax[1].imshow(crop_rgb)
    cmap = cm.get_cmap("viridis")
    for h in halos:                                   # radial wedge per sector, colour=confidence
        ix, iy = h["inner_pt"]; ox, oy = h["outer_pt"]
        ax[1].plot([ix, ox], [iy, oy], "-", color=cmap(h["confidence"] / 100.0), lw=5,
                   solid_capstyle="round")
    for s in shells:
        ax[1].contour(s["mask"].astype(float), levels=[127], colors="magenta", linewidths=1.6)
    ax[1].contour(body["mask"].astype(float), levels=[127], colors="yellow", linewidths=1.0)
    ax[1].set_title(f"spine_halo sectors (viridis=conf) + shells (magenta)")

    ax[2].imshow(crop_rgb)
    ax[2].contour(body["mask"].astype(float), levels=[127], colors="yellow", linewidths=1.4)
    if silh is not None:
        ys, xs = np.where(silh["mask"] > 0); ax[2].scatter(xs, ys, s=2, c="orange", marker="s")
    for h in halos:
        ix, iy = h["inner_pt"]; ox, oy = h["outer_pt"]
        ax[2].plot([ix, ox], [iy, oy], "-", color="cyan", lw=3)
    for s in shells:
        ax[2].contour(s["mask"].astype(float), levels=[127], colors="magenta", linewidths=1.3)
    ax[2].set_title(f"cover={roll['coverage']:.2f} ext=x{roll['mean_extent']:.2f} "
                    f"sym={roll['symmetry']:.2f} | {len(shells)} shells")

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
    # PiDiNet (shells only)
    ap.add_argument("--pidinet-weights", default="table7_pidinet.pth")
    ap.add_argument("--pidinet-device", default=None)
    # annulus / halo
    ap.add_argument("--annulus-scale", type=float, default=1.8)
    ap.add_argument("--n-sectors", type=int, default=12)
    ap.add_argument("--rays-per-sector", type=int, default=4)
    ap.add_argument("--r-steps", type=int, default=24)
    ap.add_argument("--min-contrast", type=float, default=0.04,
                    help="min B_dino edge-vs-far contrast for a sector to count")
    ap.add_argument("--strength-ref", type=float, default=0.30, help="conf: strength->100 ref")
    ap.add_argument("--extent-ref", type=float, default=1.60, help="conf: extent->100 ref")
    ap.add_argument("--falloff-ref", type=float, default=50.0, help="conf: support falloff->100 ref")
    # shells
    ap.add_argument("--shell-bright-k", type=float, default=1.0)
    ap.add_argument("--shell-min-frac", type=float, default=0.03)
    ap.add_argument("--shell-min-pidi", type=float, default=0.06)
    ap.add_argument("--max-shells", type=int, default=4)
    ap.add_argument("--max-dets", type=int, default=40)
    ap.add_argument("--out-dir", default="./m4_out")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.fresh and os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    id2file, name2cat, cat2name, _ = core.load_gt_index(args.gt_json)
    if args.class_name not in name2cat:
        raise SystemExit(f"'{args.class_name}' not in {list(name2cat)[:12]}...")
    target = name2cat[args.class_name]
    model = core.load_dinov3(args.repo_root, args.weights, args.weights_kind,
                             prefer_ema=not args.no_ema, device=device)
    edger = PiDiNetEdger(args.pidinet_weights, device=args.pidinet_device or device)
    by_img = core.load_predictions(args.pred_json, args.box_format)

    hp = dict(annulus_scale=args.annulus_scale, n_sectors=args.n_sectors,
              rays_per_sector=args.rays_per_sector, r_steps=args.r_steps,
              min_contrast=args.min_contrast, strength_ref=args.strength_ref,
              extent_ref=args.extent_ref, falloff_ref=args.falloff_ref)
    shp = dict(shell_bright_k=args.shell_bright_k, shell_min_frac=args.shell_min_frac,
               shell_min_pidi=args.shell_min_pidi, max_shells=args.max_shells)

    cov, ext, nshl, done = [], [], [], 0
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
            bdino = core.feature_boundary_map(tok, grid, connectivity=args.conn)
            bdino_up = core.upsample_grid(bdino, (ch, cw), "bilinear")
            support_up = core.upsample_grid(support, (ch, cw), "bilinear")
            obj_proto, bg_proto = core.object_bg_prototypes(
                tok, grid, mhw, dbx, (ch, cw), args.seed_frac)
            mask_u8 = (core.upsample_mask(mask_grid, (ch, cw)).astype(np.uint8)) * 255

            body = ellipse_body_atom(mask_u8, support, bdino_up, (ch, cw), tok, grid,
                                     mask_grid, obj_proto, bg_proto, ridge_snap=False)
            if body is None:
                continue
            body_feat = body.get("rfeat")
            silh_list = silhouette_contours(mask_u8, dict(silhouette_eps=0.01))
            silh = silh_list[0] if silh_list else None

            prob = edger.prob(crop_rgb)
            inner, annulus = annulus_zone(body["ellipse"], (ch, cw), dbx, args.annulus_scale)
            shells = shell_atoms(crop_rgb, body["ellipse"], inner, prob, tok, grid,
                                 support_up, body_feat, shp)
            halos, roll = halo_sector_atoms(bdino_up, support_up, body["ellipse"],
                                            (ch, cw), shells, hp)

            write_facts(os.path.join(args.out_dir, stem + ".lp"), dbx, body, silh,
                        shells, halos, roll, probe, mq, cls=args.class_name)
            visualize(crop_rgb, dbx, body, silh, shells, halos, roll, annulus,
                      os.path.join(args.out_dir, stem + ".png"),
                      title=f"{fname} det#{k} s={d['score']:.2f} bimod={probe['bimodality']:.2f} "
                            f"cover={roll['coverage']:.2f} sectors={roll['present']}/"
                            f"{roll['n_sectors']} shells={len(shells)} body_c={body_confidence(body)}")
            cov.append(roll["coverage"]); ext.append(roll["mean_extent"])
            nshl.append(len(shells)); done += 1
            if done >= args.max_dets:
                break
        if done >= args.max_dets:
            break

    print(f"\nWrote {done} decompositions to {args.out_dir}")
    if cov:
        print(f"halo_coverage : mean {np.mean(cov):.2f}  median {np.median(cov):.2f}  "
              f"(>=0.4 on {100*np.mean(np.array(cov)>=0.4):.0f}% of dets)")
        print(f"mean_extent   : mean x{np.mean(ext):.2f}")
        print(f"shells/det    : mean {np.mean(nshl):.2f}  (>=1 on "
              f"{100*np.mean(np.array(nshl)>=1):.0f}%)")


if __name__ == "__main__":
    main()