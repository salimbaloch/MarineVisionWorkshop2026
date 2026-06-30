#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m1_foundations.py  —  Milestone 1: FOUNDATION FEATURES (diagram box 2).

Visualizes, per detection, the four things DINOv3 gives us BEFORE any atom is
built, and saves them so they can be eyeballed:

    panel 1  crop + DETR box
    panel 2  object support map  S          (compute_support)
    panel 3  feature-boundary map  B_dino   (feature_boundary_map) + mask edge
    panel 4  cleaned mask                    (clean_mask)

It also computes the Step-1 per-atom feature oracle for the BODY region and
prints / overlays it:

    feature_objectness(body)   semantic "is this the object" margin   (0-100)
    region_coherence(body)     is the body one uniform material        (0-100)
    edge_on_Bdino  = ratio of B_dino on the mask perimeter vs interior

That last number is the validation you asked for: if the mask edge sits on a
real DINOv3 feature ridge the ratio is >> 1; if the mask ballooned into flat
seabed the ratio is ~1. With --qa it is correlated against mask-IoU vs the
SeaClear GT polygon, so you can confirm B_dino flags the bad masks.

Run on your machine. Deps: torch, numpy, PIL, scipy, matplotlib, core.py.
"""
import os
import argparse

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import core


def poly_to_mask(seg, h, w):
    from pycocotools import mask as coco_mask
    if not isinstance(seg, list):
        return None
    rles = coco_mask.frPyObjects(seg, h, w)
    rle = coco_mask.merge(rles) if isinstance(rles, list) else rles
    return coco_mask.decode(rle).astype(np.uint8)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def visualize(crop_np, det_box_crop, support, bdino, mask_grid, crop_hw, out_path, title):
    ch, cw = crop_hw
    sup_up = core.upsample_grid(support, crop_hw, "bilinear")
    bd_up = core.upsample_grid(bdino, crop_hw, "bilinear")
    bd_up = bd_up / (bd_up.max() + 1e-6)                 # display-normalize B_dino
    mask_up = core.upsample_mask(mask_grid, crop_hw) if mask_grid is not None \
        else np.zeros((ch, cw), bool)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    bx1, by1, bx2, by2 = det_box_crop

    ax[0].imshow(crop_np)
    ax[0].add_patch(Rectangle((bx1, by1), bx2 - bx1, by2 - by1, fill=False,
                              edgecolor="lime", linewidth=2))
    ax[0].set_title("crop + DETR box")

    ax[1].imshow(crop_np)
    im1 = ax[1].imshow(sup_up, cmap="jet", alpha=0.55, vmin=0, vmax=100)
    ax[1].set_title("object support map  S  (0-100)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].imshow(crop_np)
    im2 = ax[2].imshow(bd_up, cmap="magma", alpha=0.65, vmin=0, vmax=1)
    if mask_grid is not None:
        # the mask edge SHOULD trace the B_dino ridge; cyan = mask boundary
        ax[2].contour(mask_up.astype(float), levels=[0.5], colors="cyan", linewidths=1.5)
    ax[2].set_title("feature-boundary  B_dino  (+mask edge, cyan)")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    ax[3].imshow(crop_np)
    ov = np.zeros((ch, cw, 4)); ov[mask_up] = [1, 0, 0, 0.4]
    ax[3].imshow(ov)
    ax[3].set_title("cleaned mask (largest-CC + fill)")

    for a in ax:
        a.set_xlim(0, cw); a.set_ylim(ch, 0); a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--weights-kind", choices=["auto", "pretrained", "deim"], default="deim")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--pred-json", "--pred", dest="pred_json", required=True)
    ap.add_argument("--gt-json", "--gt", dest="gt_json", required=True)
    ap.add_argument("--img-root", "--images", dest="img_root", required=True)
    ap.add_argument("--class-name", default=None, help="optional class filter; omit for all")
    ap.add_argument("--box-format", choices=["xywh", "xyxy"], default="xywh")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--seed-frac", type=float, default=0.5)
    ap.add_argument("--sup-thr-frac", type=float, default=0.55)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--min-grid", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--conn", type=int, choices=[4, 8], default=8,
                    help="B_dino neighbour connectivity")
    ap.add_argument("--qa", action="store_true")
    ap.add_argument("--match-thr", type=float, default=0.1, help="QA: box-IoU to match a det to GT")
    ap.add_argument("--max-dets", type=int, default=40)
    ap.add_argument("--fresh", action="store_true", help="clear out-dir before writing")
    ap.add_argument("--out-dir", default="./m1_out")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.fresh and os.path.isdir(args.out_dir):
        import shutil; shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    id2file, name2cat, cat2name, gt_seg = core.load_gt_index(args.gt_json)
    id2hw = {}
    target = None
    if args.class_name:
        if args.class_name not in name2cat:
            raise SystemExit(f"'{args.class_name}' not in {list(name2cat)[:12]}...")
        target = name2cat[args.class_name]
        print(f"Target '{args.class_name}' -> id={target}")
    else:
        print("No class filter: running over all detections.")

    model = core.load_dinov3(args.repo_root, args.weights, args.weights_kind,
                             prefer_ema=not args.no_ema, device=device)
    by_img = core.load_predictions(args.pred_json, args.box_format)

    # need image H,W for QA polygon rasterization
    import json
    if args.qa:
        gtj = json.load(open(args.gt_json))
        id2hw = {im["id"]: (im["height"], im["width"]) for im in gtj["images"]}

    qa_rows = []
    done = 0
    for image_id, dets in by_img.items():
        dets = [d for d in dets if (target is None or d["cat"] == target)
                and d["score"] >= args.score_thr]
        if not dets:
            continue
        fname = id2file.get(image_id)
        if not fname:
            continue
        ipath = os.path.join(args.img_root, fname)
        if not os.path.exists(ipath):
            continue
        image_np = np.asarray(Image.open(ipath).convert("RGB"))

        gts = gt_seg.get(image_id, []) if args.qa else []
        used = [False] * len(gts)

        for k, d in enumerate(dets):
            crop_np, (ox1, oy1, _, _) = core.crop_with_margin(image_np, d["box"], args.margin)
            if crop_np.size == 0 or min(crop_np.shape[:2]) < 6:
                continue
            ch, cw = crop_np.shape[:2]
            dbx = (d["box"][0] - ox1, d["box"][1] - oy1,
                   d["box"][2] - ox1, d["box"][3] - oy1)
            ten, mhw = core.to_model_input(crop_np, args.min_grid, args.max_side, device)
            tok, grid = core.extract_tokens(model, ten, args.layer)

            support, gate = core.compute_support(tok, grid, mhw, dbx, (ch, cw), args.seed_frac)
            mask_grid = core.clean_mask(support, args.sup_thr_frac)

            # ---- STEP 1 foundation features ----
            bdino = core.feature_boundary_map(tok, grid, connectivity=args.conn)
            obj_proto, bg_proto = core.object_bg_prototypes(
                tok, grid, mhw, dbx, (ch, cw), args.seed_frac)
            body_feat = core.region_feature(tok, grid, mask_grid)
            objness = core.feature_objectness(body_feat, obj_proto, bg_proto)
            coher = core.region_coherence(tok, grid, mask_grid)
            edge = core.mask_edge_on_bdino(bdino, mask_grid)
            mask_frac = float(mask_grid.mean()) if mask_grid is not None else 0.0

            if args.qa:
                bj, best = -1, args.match_thr
                for j, g in enumerate(gts):
                    if used[j] or (target is not None and g["cat"] != target):
                        continue
                    a, b = d["box"], g["box"]
                    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
                    iou = inter / ua if ua > 0 else 0.0
                    if iou >= best:
                        best, bj = iou, j
                miou = np.nan
                if bj >= 0 and mask_grid is not None:
                    used[bj] = True
                    H, W = id2hw[image_id]
                    gmask = poly_to_mask(gts[bj]["seg"], H, W)
                    if gmask is not None:
                        full = np.zeros((H, W), bool)
                        mu = core.upsample_mask(mask_grid, (ch, cw))
                        full[oy1:oy1+ch, ox1:ox1+cw] = mu
                        miou = mask_iou(full, gmask > 0)
                qa_rows.append(dict(gate=gate, mask_frac=mask_frac,
                                    objness=objness, coher=coher,
                                    edge_ratio=edge["ratio"],
                                    sharp=core.mask_quality(support, mask_grid)["boundary_sharpness"],
                                    score=d["score"], miou=miou))
            else:
                stem = f"{os.path.splitext(os.path.basename(fname))[0]}_det{k}"
                visualize(crop_np, dbx, support, bdino, mask_grid, (ch, cw),
                          os.path.join(args.out_dir, stem + ".png"),
                          title=f"{fname} det#{k} s={d['score']:.2f} grid={grid[0]}x{grid[1]} "
                                f"gate={gate:.2f} mask={mask_frac:.2f}  |  "
                                f"objness={objness:.0f} coher={coher:.0f} "
                                f"edge_on_Bdino={edge['ratio']:.2f}")
            done += 1
            if done >= args.max_dets:
                break
        if done >= args.max_dets:
            break

    if args.qa:
        qa_report(qa_rows, args.sup_thr_frac)
    else:
        print(f"\nWrote {done} foundation-feature panels to {args.out_dir}")


def qa_report(rows, thr):
    if not rows:
        print("No detections."); return
    a = lambda k: np.array([r[k] for r in rows], float)
    miou = a("miou"); miou = miou[~np.isnan(miou)]
    print(f"\n===== M1 FOUNDATION QA  ({len(rows)} detections, thr={thr}) =====")
    for k, lab in [("gate", "gate"), ("mask_frac", "mask coverage of crop"),
                   ("objness", "feature_objectness (body)"),
                   ("coher", "region_coherence (body)"),
                   ("edge_ratio", "B_dino edge ratio (perim/interior)"),
                   ("sharp", "boundary sharpness (support)")]:
        v = a(k); print(f"  {lab:34} mean {v.mean():7.3f}  median {np.median(v):7.3f}")
    print(f"  {'low gate <0.5':34} {100*(a('gate')<0.5).mean():6.1f}%")
    print(f"  {'ballooned mask >0.55 crop':34} {100*(a('mask_frac')>0.55).mean():6.1f}%")
    if miou.size:
        print(f"\n  mask-IoU vs GT polygon: mean {miou.mean():.3f}  median {np.median(miou):.3f}  "
              f"%>=.5 {100*(miou>=.5).mean():.0f}  %>=.7 {100*(miou>=.7).mean():.0f}  (n={miou.size})")
        # THE validation: does B_dino edge ratio predict mask-IoU? split at its median.
        er = a("edge_ratio"); valid = ~np.isnan(a("miou"))
        thr_er = np.median(er[valid])
        print(f"\n  mask-IoU split at median B_dino edge ratio ({thr_er:.2f}):")
        for lab, sel in [("ridge half  (edge>=med)", valid & (er >= thr_er)),
                         ("flat half   (edge< med)", valid & (er < thr_er))]:
            if sel.sum():
                mi = a("miou")[sel]
                print(f"    {lab:24} n={int(sel.sum()):4d}  mask-IoU mean {mi.mean():.3f}  "
                      f"med {np.median(mi):.3f}")
        print("    (if the ridge half has clearly higher IoU, B_dino discriminates good")
        print("     masks from ballooned ones -> it is a usable mask-quality referee.)")
    print("\n  reading: objness should be HIGH on a clean urchin body, LOW where the")
    print("  mask leaked onto seabed; edge ratio >>1 means the mask edge sits on a")
    print("  real DINOv3 feature boundary, ~1 means it ballooned into a flat field.")


if __name__ == "__main__":
    main()