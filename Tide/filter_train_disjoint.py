#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_train_disjoint.py -- remove the SeaClear cross-split frame leak.

SeaClear's official train/val split scatters consecutive video frames across
both sides (~84% of val images have a near-identical train frame). If the
refinement (ASP weights + regressor) is trained on raw train detections and
tested on val, it is partly tested on near-duplicates of its training frames.

Fix implemented here: compute a sequence key for every training stem and every
val stem (same convention as make_split.py: image basename with the trailing
frame number stripped, or --seq-regex group(1)), then build a symlinked copy of
the TRAIN decomposition that EXCLUDES every train sequence whose key also
occurs in val. Val stays intact, so refined AP remains comparable to the
detector's val AP.

Output layout (drop-in for learn_weights / alternate_train / box_regressor):
    <out-root>/lp/*.lp   <out-root>/emb/*.npz   <out-root>/gt_boxes.json
    <out-root>/crop_meta.json

Usage:
    python filter_train_disjoint.py \
      --train-lp-dir $OUT/urchin/dec_train --train-emb-dir $OUT/urchin/dec_train/emb \
      --train-gt $OUT/urchin/dec_train/gt_boxes.json \
      --train-crop-meta $OUT/urchin/dec_train/crop_meta.json \
      --val-lp-dir $OUT/urchin/dec \
      --out-root $OUT/urchin/train_disjoint
"""
import os, re, glob, json, argparse
from collections import defaultdict


def seq_key(stem, regex=None):
    imgbase = stem.rsplit("_det", 1)[0]
    if regex:
        m = re.search(regex, imgbase)
        return m.group(1) if m else imgbase
    return re.sub(r"[_-]?\d+$", "", imgbase) or imgbase


def link(src, dst, copy=False):
    if os.path.lexists(dst):
        os.remove(dst)
    if copy:
        import shutil; shutil.copy(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def stems_of(lp_dir):
    return [os.path.splitext(os.path.basename(p))[0]
            for p in sorted(glob.glob(os.path.join(lp_dir, "*.lp")))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-lp-dir", required=True)
    ap.add_argument("--train-emb-dir", default=None, help="default <train-lp-dir>/emb")
    ap.add_argument("--train-gt", required=True)
    ap.add_argument("--train-crop-meta", default=None)
    ap.add_argument("--val-lp-dir", required=True,
                    help="val decomposition dir; its stems define the forbidden sequences")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seq-regex", default=None)
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    emb_dir = args.train_emb_dir or os.path.join(args.train_lp_dir, "emb")
    gt = json.load(open(args.train_gt))
    cmeta = (json.load(open(args.train_crop_meta))
             if args.train_crop_meta and os.path.exists(args.train_crop_meta) else {})

    val_keys = {seq_key(s, args.seq_regex) for s in stems_of(args.val_lp_dir)}
    if not val_keys:
        raise SystemExit(f"no .lp stems found in --val-lp-dir {args.val_lp_dir}")

    tr_stems = [s for s in stems_of(args.train_lp_dir) if s in gt]
    groups = defaultdict(list)
    for s in tr_stems:
        groups[seq_key(s, args.seq_regex)].append(s)

    keep_keys = sorted(k for k in groups if k not in val_keys)
    drop_keys = sorted(k for k in groups if k in val_keys)
    keep = [s for k in keep_keys for s in groups[k]]
    drop_n = sum(len(groups[k]) for k in drop_keys)

    print(f"train stems={len(tr_stems)}  sequences={len(groups)}  val sequences={len(val_keys)}")
    print(f"COLLIDING train sequences dropped: {len(drop_keys)} ({drop_n} stems)")
    print(f"KEPT: {len(keep_keys)} sequences ({len(keep)} stems)")
    if drop_keys[:5]:
        print("example dropped keys:", ", ".join(drop_keys[:5]))
    print("example kept  stem -> key:")
    for s in keep[:8]:
        print(f"    {s:40s} -> {seq_key(s, args.seq_regex)}")
    if not keep:
        raise SystemExit("EVERYTHING collided -- check --seq-regex; keys may be too coarse")
    if drop_n == 0:
        print("WARNING: zero collisions found. Given SeaClear's known ~84% frame overlap, "
              "this most likely means the sequence key is too fine (every frame its own key). "
              "Inspect the printed keys and set --seq-regex.")

    lp_o = os.path.join(args.out_root, "lp"); em_o = os.path.join(args.out_root, "emb")
    os.makedirs(lp_o, exist_ok=True); os.makedirs(em_o, exist_ok=True)
    for s in keep:
        link(os.path.join(args.train_lp_dir, s + ".lp"), os.path.join(lp_o, s + ".lp"), args.copy)
        ep = os.path.join(emb_dir, s + ".npz")
        if os.path.exists(ep):
            link(ep, os.path.join(em_o, s + ".npz"), args.copy)
    json.dump({s: gt[s] for s in keep}, open(os.path.join(args.out_root, "gt_boxes.json"), "w"))
    if cmeta:
        json.dump({s: cmeta[s] for s in keep if s in cmeta},
                  open(os.path.join(args.out_root, "crop_meta.json"), "w"))
    print(f"wrote {args.out_root}/{{lp,emb,gt_boxes.json,crop_meta.json}}")


if __name__ == "__main__":
    main()