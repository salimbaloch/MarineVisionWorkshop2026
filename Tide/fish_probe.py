#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fish_probe.py  —  LEAN fish decomposition that DOUBLES AS THE OPTION-B PROBE.

Goal: produce a valid three-stage .lp for fish AND, in the same run, the numbers
that decide whether Option B (multi-part decomposition: body + fins + tail) is
real for this dataset. We do NOT cluster parts here. We measure whether the
DINOv3 support even HOLDS the fins, and whether the inside is feature-separable.

Differences from decompose.py (urchin), all because a fish is NOT one blob:
  * BODY REGION = the support MASK (not the fitted ellipse). An ellipse would
    crop the fins away and HIDE the very thing we're testing for. So Stage-1 r1
    geometry (centroid/bbox/area) comes from the mask; the ellipse is kept only
    as a secondary geometric primitive.
  * BOUNDARY: silhouette is PRIMARY (b1), ellipse SECONDARY (b2). Fish outlines
    are not elliptical.
  * Foreign objects OFF, spines OFF.

OPTION-B DIAGNOSTICS (printed + in panel titles), per detection:
  bimodality   feature_variance_probe: are inside tokens multi-material? (the
               routing number -- high => parts separable by FEATURE => B viable)
  dispersion   feature spread inside the mask
  solidity     area / convex_hull_area : LOW => protrusions held (fins/tail) =>
               parts present GEOMETRICALLY. HIGH => clean blob => no fins held.
  elongation   major/minor axis (fish are elongated)
  n_defects    concave notches between protrusions (gaps between fins)
  fit_error    ellipse-vs-mask (HIGH => non-elliptical => fins stick out)
  mask_cov     mask area / crop area
  fins_held    proxy = (solidity < solid_thr) or (n_defects >= 2)

READING IT (the decision, which the run only INFORMS):
  * many fins_held AND high bimodality -> parts are there AND separable -> build
    Option B (feature clustering into body/fin/tail).
  * fins_held but LOW bimodality -> fins held geometrically but not feature-
    distinct -> clustering won't label them; geometry-split only.
  * mostly HIGH solidity (clean blobs) -> fins washed out -> Option B dead here,
    fish collapses to single-region (Option A).
  Expect a MIXED result (fins held on clear fish, washed on turbid). That's not
  failure -- it's the router's job: parts when the features support them.

Emits routing facts into each .lp (bimodality / region_solidity /
region_elongation / concavities) so the future Option-B router can read them.

Run on your machine. Reuses decompose.py helpers. Deps: torch, numpy, cv2, PIL,
scipy, matplotlib, core.py, m2_regions.py, decompose.py (+ its imports),
pidinet_edge.py + table7_pidinet.pth.
"""
import os
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
from m2_regions import ellipse_body_atom, ellipse_to_mask, shape_descriptors
import decompose as D          # reuse the frozen three-stage helpers


# ============================================================ fish facts (mask-region body)
def write_facts_fish(path, det_box_crop, region, geoms, boundaries, relations,
                     diag, oid="o1", cls="animal_fish"):
    L = ["% fish decomposition (spec stages 1-2-3). SILHOUETTE-PRIMARY body.",
         "% body region = support MASK (holds fins), NOT a fitted ellipse.",
         "% foreign objects OFF, spines OFF. Routing diagnostics emitted for Option B.",
         ""]
    bx = [int(round(v)) for v in det_box_crop]
    L += [f"detection({oid}).", f"class({oid},{cls}).",
          f"proposal_box({oid},box({bx[0]},{bx[1]},{bx[2]},{bx[3]}))."]

    # routing diagnostics (detection-level) -- what an Option-B router will read
    L += ["", "% --- routing diagnostics (for Option B) ---",
          f"bimodality({oid},{int(round(diag['bimodality']*100))}).",
          f"feature_dispersion({oid},{int(round(diag['dispersion']*100))}).",
          f"region_solidity({oid},{int(round(diag['solidity']*100))}).",
          f"region_elongation({oid},{int(round(diag['elongation']*100))}).",
          f"concavities({oid},{diag['n_defects']}).",
          f"ellipse_fit_error({oid},{int(round(diag['fit_error']*100))})."]
    if diag["fins_held"]:
        L.append(f"fins_held_candidate({oid}).")
    if diag["bimodality"] >= diag["bimod_thr"]:
        L.append(f"multi_material_candidate({oid}).")

    # ---------- STAGE 1: semantic region (body = mask) ----------
    L += ["", "% --- Stage 1: semantic region (body = support mask) ---"]
    rcx, rcy = [int(round(v)) for v in region["centroid"]]
    rb = [int(round(v)) for v in region["bbox"]]
    L += [f"semantic_region(r1).", f"belongs_to_detection(r1,{oid}).",
          f"region_role(r1,body).",
          f"region_centroid(r1,point({rcx},{rcy})).",
          f"region_box(r1,box({rb[0]},{rb[1]},{rb[2]},{rb[3]})).",
          f"region_area(r1,{region['area']}).",
          f"semantic_support(r1,{int(round(region['support']))}).",
          f"region_objectness(r1,{int(round(region['objness']))}).",
          f"region_coherence(r1,{int(round(region['coher']))}).",
          f"inside(r1,{oid})."]

    # ---------- STAGE 2: geometric primitives ----------
    L += ["", "% --- Stage 2: geometric primitives ---"]
    ecx, ecy, ed1, ed2, eang = region["ellipse"]
    eb = [int(round(v)) for v in region["ellipse_bbox"]]
    L += [f"geometric_primitive(g1).", f"belongs_to_detection(g1,{oid}).",
          f"primitive_type(g1,ellipse).",
          f"ellipse(g1,point({int(round(ecx))},{int(round(ecy))}),"
          f"axes({int(round(ed1))},{int(round(ed2))}),angle({int(round(eang))})).",
          f"primitive_box(g1,box({eb[0]},{eb[1]},{eb[2]},{eb[3]})).",
          f"fit_error(g1,{int(round(diag['fit_error']*100))}).",
          f"edge_confidence(g1,{int(round(region.get('ellipse_edge_conf',0)))}).",
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

    # ---------- STAGE 3: boundary primitives (silhouette primary) ----------
    L += ["", "% --- Stage 3: boundary primitives (silhouette b1 primary, ellipse b2) ---"]
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

    L += ["", "% --- relations ---"]
    for r, x, y in relations:
        L.append(f"{r}({x},{y}).")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ============================================================ visualize (fins-focused)
def visualize(crop_rgb, det_box_crop, support_up, mask_u8, region, geoms,
              boundaries, diag, out_path, title):
    H, W = crop_rgb.shape[:2]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    bx1, by1, bx2, by2 = det_box_crop

    # 1: crop + box + mask
    ax[0].imshow(crop_rgb)
    ax[0].add_patch(Rectangle((bx1, by1), bx2-bx1, by2-by1, fill=False, ec="lime", lw=2))
    ov = np.zeros((H, W, 4)); ov[mask_u8 > 0] = [1, 1, 1, 0.22]; ax[0].imshow(ov)
    ax[0].set_title("crop + DETR box + support mask")

    # 2: support map S -- THE panel: does support reach the fins?
    im = ax[1].imshow(support_up, cmap="jet", vmin=0, vmax=100)
    ax[1].set_title("support S  (do fins/tail light up?)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    # 3: fins-held read: mask (white) vs convex hull (cyan) vs ellipse (lime)
    ax[2].imshow(crop_rgb)
    ax[2].imshow(ov)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c).reshape(-1, 2)
        ax[2].plot(np.r_[hull[:, 0], hull[0, 0]], np.r_[hull[:, 1], hull[0, 1]],
                   "--", color="cyan", lw=1.3, label="convex hull")
    em = ellipse_to_mask((H, W), region["ellipse"], 1.0)
    ax[2].contour(em.astype(float), [127], colors="lime", linewidths=1.1)
    ax[2].set_title(f"mask vs hull(cyan)/ellipse(lime)  solidity={diag['solidity']:.2f} "
                    f"ndef={diag['n_defects']}")

    # 4: Stage 2/3 primitives + boundary
    ax[3].imshow(crop_rgb)
    for g in geoms:
        ys, xs = np.where(g["mask"] > 0)
        cc = "cyan" if g["location"] == "outer" else "orange"
        ax[3].scatter(xs, ys, s=3, c=cc, marker="s")
    _bcol = {"silhouette": "orange", "ellipse": "lime", "edge": "yellow"}
    for b in boundaries:
        ys, xs = np.where(b["mask"] > 0)
        ax[3].scatter(xs, ys, s=2, c=_bcol.get(b["source"], "yellow"), marker="s")
    ax[3].set_title(f"Stage 2/3: boundary={len(boundaries)} (silhouette primary)")

    for a in ax:
        a.set_xlim(0, W); a.set_ylim(H, 0); a.axis("off")
    ax[1].axis("on"); ax[1].set_xticks([]); ax[1].set_yticks([])
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
    ap.add_argument("--class-name", default="animal_fish")
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
    # location / boundary
    ap.add_argument("--perim-px", type=int, default=3)
    ap.add_argument("--on-perim-min", type=float, default=0.5)
    ap.add_argument("--inside-min", type=float, default=0.6)
    ap.add_argument("--silhouette-eps", type=float, default=0.01)
    ap.add_argument("--edge-conf-min", type=float, default=35.0)
    # relations
    ap.add_argument("--parallel-deg", type=float, default=12.0)
    ap.add_argument("--collinear-px", type=float, default=4.0)
    # Option-B thresholds (diagnostic only)
    ap.add_argument("--bimod-thr", type=float, default=2.2,
                    help="bimodality >= this -> multi_material_candidate (router go-signal)")
    ap.add_argument("--solid-thr", type=float, default=0.88,
                    help="solidity < this (or >=2 concavities) -> fins_held_candidate")
    # io
    ap.add_argument("--max-dets", type=int, default=10)
    ap.add_argument("--out-dir", default="./fish_probe")
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

    rows, done = [], 0
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

            # body atom (for objness/coher/rfeat/ellipse); region geometry from MASK
            body = ellipse_body_atom(mask_u8, support, None, (ch, cw), tok, grid,
                                     mask_grid, obj_proto, bg_proto, ridge_snap=False)
            if body is None:
                continue
            desc = shape_descriptors(mask_u8)
            if desc is None:
                continue

            # ---- Option-B diagnostics ----
            probe = core.feature_variance_probe(tok, grid, mask_grid)
            fins_held = (desc["solidity"] < args.solid_thr) or (desc["n_defects"] >= 2)
            diag = dict(bimodality=probe["bimodality"], dispersion=probe["dispersion"],
                        solidity=desc["solidity"], elongation=desc["elongation"],
                        n_defects=desc["n_defects"], fit_error=body["fit_error"],
                        mask_cov=float((mask_u8 > 0).mean()),
                        fins_held=fins_held, bimod_thr=args.bimod_thr)

            # ---- region (body = MASK, not ellipse) ----
            region = dict(centroid=desc["centroid"], bbox=desc["bbox"],
                          area=int((mask_u8 > 0).sum()),
                          support=float(support_up[mask_u8 > 0].mean()),
                          objness=body["objness"], coher=body["coher"],
                          ellipse=body["ellipse"], ellipse_bbox=body["bbox"])

            prob = edger.prob(crop_rgb)

            # ---- Stage 2: geometric primitives, classified ----
            geoms = D.geometric_primitives(prob, mask_u8, gp)
            ker = np.ones((3, 3), np.uint8)
            mbin = (mask_u8 > 0).astype(np.uint8)
            interior_band = cv2.erode(mbin, ker, iterations=args.perim_px) > 0
            perim_band = (cv2.dilate(mbin, ker, iterations=args.perim_px) -
                          cv2.erode(mbin, ker, iterations=args.perim_px)) > 0
            for gi, g in enumerate(geoms, 2):
                g["id"] = f"g{gi}"
                loc, _ = D.classify_location(g["mask"], perim_band, interior_band,
                                             args.on_perim_min, args.inside_min)
                g["location"] = loc
                g["semantic_support"] = D.semantic_support_along(g["mask"], support_up)
            region["ellipse_edge_conf"] = float(prob[perim_band].mean()) * 100 if perim_band.any() else 0.0

            # ---- Stage 3: silhouette PRIMARY (b1), ellipse SECONDARY (b2) ----
            silh = D.silhouette_boundary(mask_u8, support_up, prob, args.silhouette_eps)
            ellipse_b = D.ellipse_boundary(body, support_up, prob, (ch, cw),
                                           thickness=max(2, args.perim_px))
            boundaries = D.build_boundaries(silh, ellipse_b, geoms, bp)   # silh first => b1

            relations = D.boundary_relations(boundaries, "r1", rp)

            write_facts_fish(os.path.join(args.out_dir, stem + ".lp"), dbx, region,
                             geoms, boundaries, relations, diag, cls=args.class_name)
            visualize(crop_rgb, dbx, support_up, mask_u8, region, geoms, boundaries, diag,
                      os.path.join(args.out_dir, stem + ".png"),
                      title=f"{fname} det#{k} s={d['score']:.2f} | bimod={probe['bimodality']:.2f} "
                            f"solidity={desc['solidity']:.2f} elong={desc['elongation']:.2f} "
                            f"ndef={desc['n_defects']} fins_held={fins_held}")
            rows.append(dict(name=stem, score=d["score"], **diag))
            done += 1
            if done >= args.max_dets:
                break
        if done >= args.max_dets:
            break

    report(rows, args)


def report(rows, args):
    if not rows:
        print("No fish detections."); return
    a = lambda k: np.array([r[k] for r in rows], float)
    print(f"\n===== FISH OPTION-B PROBE  ({len(rows)} detections) =====\n")
    print(f"  {'name':18} {'score':>5} | {'bimod':>6} {'disp':>5} | "
          f"{'solid':>5} {'elong':>5} {'ndef':>4} {'fit_err':>7} | {'fins_held':>9}")
    for r in rows:
        print(f"  {r['name'][:18]:18} {r['score']:5.2f} | {r['bimodality']:6.2f} "
              f"{r['dispersion']:5.2f} | {r['solidity']:5.2f} {r['elongation']:5.2f} "
              f"{r['n_defects']:4d} {r['fit_error']:7.2f} | {str(r['fins_held']):>9}")
    nb = int((a('bimodality') >= args.bimod_thr).sum())
    nf = int(sum(r['fins_held'] for r in rows))
    print(f"\n  AGGREGATE ({len(rows)} fish):")
    print(f"    bimodality            mean {a('bimodality').mean():.2f}  "
          f">= {args.bimod_thr} on {nb}/{len(rows)}  (feature-separable parts)")
    print(f"    solidity              mean {a('solidity').mean():.2f}  "
          f"(low => fins held geometrically)")
    print(f"    fins_held proxy       {nf}/{len(rows)} detections")
    print(f"    ellipse fit_error     mean {a('fit_error').mean():.2f}  "
          f"(high => non-elliptical => fins stick out)")
    print("\n  DECISION CUES:")
    print("    * many fins_held AND high bimodality -> parts are real AND separable -> build")
    print("      Option B (cluster DINO tokens into body/fin/tail).")
    print("    * fins_held but LOW bimodality -> fins held but not feature-distinct ->")
    print("      clustering won't label them; only geometry could split them.")
    print("    * mostly HIGH solidity (clean blobs) -> fins washed out -> Option B dead here,")
    print("      fish stays single-region (Option A).")
    print("    Look at panel 2 (support S): do the fins/tail actually light up, or does")
    print("    support stop at the solid body? That eyeball + these numbers = the verdict.")


if __name__ == "__main__":
    main()