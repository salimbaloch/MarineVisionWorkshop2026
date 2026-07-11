#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_labels.py -- COCO-exact TP/FP labels for the re-scoring channel.

Replaces presence-in-gt_boxes.json as the rescore label. gt_boxes.json was
built for the REGRESSOR (best GT per proposal at IoU>=0.3, non-exclusive);
as a TP/FP label it is wrong twice: (a) IoU 0.3-0.5 detections are labeled
positive but are FP at every COCO threshold, (b) two detections on one GT are
both labeled positive but COCO scores the loser as FP.

This script replicates COCOeval's matching exactly -- score-descending
exclusive greedy over the FULL prediction set (a detection's duplicate status
depends on higher-scored dets, so matching must never be run on a filtered
subset) -- and emits per detection:

    best_iou     max IoU to any same-class non-crowd GT (assignment-free)
    match_iou    IoU of the exclusive match at 0.50 (0.0 if unmatched)
    matched50    exclusive match flag at IoU 0.50   <- the binary rescore label
    matched75    exclusive match flag at IoU 0.75 (independent matching pass)
    duplicate50  best_iou >= 0.5 but lost the greedy assignment (learned-NMS
                 territory: explanation features are blind to these, context
                 features are not)

Outputs:
  --out         {"meta": {...},
                 "by_key":  {"cat|img|x1|y1|x2|y2": {...}},        always
                 "by_stem": {stem: {...}}}                          iff --crop-meta
  --stems-out   plain JSON list of stems with matched50 -- DROP-IN replacement
                for rescore.py --train-labels / --val-labels (it does
                set(json.load(...)); presence = positive).

The stem join uses rescore.py's exact key construction:
    (cat_id, image_id, round(x1,2), round(y1,2), round(x2,2), round(y2,2))
with crop_meta.json boxes already xyxy and preds boxes converted xywh->xyxy,
so joined labels line up 1:1 with what rescore extracts. Join diagnostics are
printed; unjoined stems mean key drift and should be investigated, not ignored.

Usage (SeaClear urchin, val):
  python make_labels.py \
      --preds preds_seaclear_val.json \
      --gt    /srv/data1/Salim/Underwater/DEIMv2/dataset/annotations/instances_val.json \
      --class-name animal_urchin \
      --crop-meta $DEC_VAL/crop_meta.json \
      --out $OUT/urchin/labels_val.json \
      --stems-out $OUT/urchin/matched50_val.json

Deps: numpy, pycocotools.
"""
import io
import json
import argparse
import contextlib
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO


# ------------------------------------------------------------------ matching
def _iou_mat(D, G):
    """IoU matrix for xywh boxes. D (N,4), G (M,4) -> (N,M)."""
    if len(D) == 0 or len(G) == 0:
        return np.zeros((len(D), len(G)), np.float64)
    dx1, dy1 = D[:, 0], D[:, 1]
    dx2, dy2 = D[:, 0] + D[:, 2], D[:, 1] + D[:, 3]
    gx1, gy1 = G[:, 0], G[:, 1]
    gx2, gy2 = G[:, 0] + G[:, 2], G[:, 1] + G[:, 3]
    ix1 = np.maximum(dx1[:, None], gx1[None, :])
    iy1 = np.maximum(dy1[:, None], gy1[None, :])
    ix2 = np.minimum(dx2[:, None], gx2[None, :])
    iy2 = np.minimum(dy2[:, None], gy2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aD = (D[:, 2] * D[:, 3])[:, None]
    aG = (G[:, 2] * G[:, 3])[None, :]
    return inter / np.maximum(aD + aG - inter, 1e-9)


def greedy_match(iou, scores, thr):
    """COCO-style exclusive greedy (score-descending, stable). Returns
    (match_iou per det, matched flag per det)."""
    order = np.argsort(-scores, kind="mergesort")
    taken = np.zeros(iou.shape[1], bool)
    miou = np.zeros(len(scores), np.float64)
    flag = np.zeros(len(scores), bool)
    for i in order:
        if not iou.shape[1]:
            break
        row = np.where(taken, -1.0, iou[i])
        j = int(row.argmax())
        if row[j] >= thr:
            taken[j] = True
            miou[i] = iou[i, j]
            flag[i] = True
    return miou, flag


def det_key(cat_id, image_id, x1, y1, x2, y2):
    """rescore.py's exact key, as a string for JSON."""
    return (f"{int(cat_id)}|{int(image_id)}|{round(x1, 2):.2f}|{round(y1, 2):.2f}"
            f"|{round(x2, 2):.2f}|{round(y2, 2):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--class-name", default=None)
    ap.add_argument("--cat-id", type=int, default=None)
    ap.add_argument("--crop-meta", default=None,
                    help="crop_meta.json from m4decompose -> adds by_stem + enables --stems-out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems-out", default=None,
                    help="list of matched50 stems; drop-in for rescore.py --*-labels")
    args = ap.parse_args()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cocoGt = COCO(args.gt)

    if args.cat_id is not None:
        cat = args.cat_id
        cname = cocoGt.cats.get(cat, {}).get("name", str(cat))
    elif args.class_name:
        hits = [c["id"] for c in cocoGt.cats.values() if c["name"] == args.class_name]
        if not hits:
            raise SystemExit(f"'{args.class_name}' not in "
                             f"{[c['name'] for c in cocoGt.cats.values()][:15]}")
        cat, cname = hits[0], args.class_name
    else:
        raise SystemExit("need --class-name or --cat-id")

    preds = json.load(open(args.preds))
    img_ids = set(cocoGt.getImgIds())
    n0 = len(preds)
    preds = [p for p in preds if p["image_id"] in img_ids]
    if len(preds) < n0:
        print(f"[warn] dropped {n0 - len(preds)} predictions not in the GT split")

    tgt = [p for p in preds if p["category_id"] == cat]
    if not tgt:
        raise SystemExit(f"no predictions with category_id={cat}")

    gt_by_img = defaultdict(list)
    n_crowd = 0
    for a in cocoGt.loadAnns(cocoGt.getAnnIds(catIds=[cat])):
        if a.get("iscrowd", 0):
            n_crowd += 1
            continue
        gt_by_img[a["image_id"]].append(a["bbox"])
    n_gt = sum(len(g) for g in gt_by_img.values())
    if n_crowd:
        print(f"[warn] {n_crowd} iscrowd GTs excluded")

    by_img = defaultdict(list)
    for p in tgt:
        by_img[p["image_id"]].append(p)

    by_key, collisions = {}, 0
    n_m50 = n_m75 = n_dup = 0
    for img, dets in by_img.items():
        D = np.array([d["bbox"] for d in dets], np.float64)
        S = np.array([d["score"] for d in dets], np.float64)
        G = np.array(gt_by_img.get(img, []), np.float64).reshape(-1, 4)
        iou = _iou_mat(D, G)
        best = iou.max(1) if iou.shape[1] else np.zeros(len(dets))
        mi50, f50 = greedy_match(iou, S, 0.50)
        _, f75 = greedy_match(iou, S, 0.75)          # independent pass, like COCO
        for k, d in enumerate(dets):
            x, y, w, h = d["bbox"]
            key = det_key(cat, img, x, y, x + w, y + h)
            rec = dict(score=float(d["score"]),
                       best_iou=round(float(best[k]), 4),
                       match_iou=round(float(mi50[k]), 4),
                       matched50=bool(f50[k]),
                       matched75=bool(f75[k]),
                       duplicate50=bool(best[k] >= 0.5 and not f50[k]),
                       image_id=int(img))
            if key in by_key:
                collisions += 1
            by_key[key] = rec
            n_m50 += rec["matched50"]; n_m75 += rec["matched75"]
            n_dup += rec["duplicate50"]

    print(f"[{cname}] {len(tgt)} detections / {n_gt} GTs   "
          f"matched50 {n_m50}   matched75 {n_m75}   duplicate50 {n_dup}")
    if collisions:
        print(f"[warn] {collisions} identical (cat,img,box) keys -- later det overwrote earlier")

    # ------------------------------------------------------------ stem join
    by_stem = {}
    if args.crop_meta:
        cmeta = json.load(open(args.crop_meta))
        unjoined = []
        for stem, cm in cmeta.items():
            key = det_key(cat, cm[0], cm[3], cm[4], cm[5], cm[6])
            if key in by_key:
                by_stem[stem] = by_key[key]
            else:
                unjoined.append(stem)
        print(f"joined {len(by_stem)}/{len(cmeta)} crop_meta stems to predictions")
        if unjoined:
            print(f"[WARN] {len(unjoined)} stems did NOT join (key drift?) -- e.g. "
                  + ", ".join(unjoined[:5]))
            print("       investigate before training; unjoined stems get no label.")

    meta = dict(cls=cname, cat_id=int(cat), n_dets=len(tgt), n_gt=n_gt,
                n_matched50=int(n_m50), n_matched75=int(n_m75),
                n_duplicate50=int(n_dup), preds=args.preds, gt=args.gt)
    json.dump(dict(meta=meta, by_key=by_key, by_stem=by_stem),
              open(args.out, "w"))
    print(f"wrote {args.out}")

    if args.stems_out:
        if not args.crop_meta:
            raise SystemExit("--stems-out needs --crop-meta (labels are keyed by stem)")
        stems = sorted(s for s, r in by_stem.items() if r["matched50"])
        json.dump(stems, open(args.stems_out, "w"))
        print(f"wrote {args.stems_out}  ({len(stems)} matched50 stems; drop-in for "
              f"rescore.py --*-labels)")


if __name__ == "__main__":
    main()