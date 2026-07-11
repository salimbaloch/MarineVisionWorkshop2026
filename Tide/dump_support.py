#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_support.py -- persist the DINOv3 support map per detection (crop frame),
so localize_ceiling.py --support-dir can sweep support thresholds for candidate
boxes. Mirrors m4decompose's crop loop EXACTLY (same margin / min-grid /
max-side / layer / seed-frac / score-thr / stem naming), so the dumped maps
align 1:1 with the existing .lp boxes and gt_boxes.json. Skips PiDiNet, ellipse
fitting, B_dino, ASP -- support only, so it is much faster than full
decomposition. Val-only (a few thousand crops) is the intended use.

Saves <out-dir>/<stem>.npz with key 'sup' = uint8 (ch x cw), support*255.

Usage (SeaClear urchin val, matching your decompose knobs):
  python dump_support.py --profile urchin --class-name animal_urchin \
    --repo-root $REPO --weights $WEIGHTS \
    --pred-json $PRED_VAL --gt-json $GT_VAL --img-root $IMG_VAL \
    --box-format xywh --score-thr 0.05 --out-dir $OUT/urchin/dec_val/support

Then:
  python localize_ceiling.py ... --support-dir $OUT/urchin/dec_val/support
"""
import os, argparse
import numpy as np
from PIL import Image

import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-name", required=True)
    ap.add_argument("--profile", default=None, help="unused; accepted for CLI parity")
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--weights-kind", default="deim")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--pred-json", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--box-format", choices=["xywh", "xyxy"], default="xywh")
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--seed-frac", type=float, default=0.5)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--min-grid", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-dets", type=int, default=100000)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    id2file, name2cat, cat2name, _ = core.load_gt_index(args.gt_json)
    if args.class_name not in name2cat:
        raise SystemExit(f"'{args.class_name}' not in {list(name2cat)[:12]}...")
    target = name2cat[args.class_name]
    print(f"[support] target '{args.class_name}' -> id={target}  out={args.out_dir}")

    model = core.load_dinov3(args.repo_root, args.weights, args.weights_kind,
                             prefer_ema=not args.no_ema, device=device)
    by_img = core.load_predictions(args.pred_json, args.box_format)

    n_done = n_skip = 0
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
                n_skip += 1; continue
            ch, cw = crop_rgb.shape[:2]
            stem = f"{os.path.splitext(os.path.basename(fname))[0]}_det{k}"
            dbx = (d["box"][0]-ox1, d["box"][1]-oy1, d["box"][2]-ox1, d["box"][3]-oy1)
            ten, mhw = core.to_model_input(crop_rgb, args.min_grid, args.max_side, device)
            tok, grid = core.extract_tokens(model, ten, args.layer)
            support, _ = core.compute_support(tok, grid, mhw, dbx, (ch, cw), args.seed_frac)
            sup_up = core.upsample_grid(support, (ch, cw), "bilinear")
            sup_up = np.clip(sup_up, 0.0, 1.0)
            np.savez_compressed(os.path.join(args.out_dir, stem + ".npz"),
                                sup=(sup_up * 255).astype(np.uint8))
            n_done += 1
            if n_done % 500 == 0:
                print(f"    {n_done} support maps dumped...")
            if n_done >= args.max_dets:
                break
        if n_done >= args.max_dets:
            break
    print(f"[support] dumped {n_done} maps ({n_skip} tiny crops skipped) -> {args.out_dir}")


if __name__ == "__main__":
    main()