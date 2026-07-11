#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_coco_ap.py -- end-to-end pipeline evaluation on a full COCO split.

Runs the trained pipeline the paper describes (Sec 3.3-3.4): for every target-
class detection, ASP selects the explanation under the learned weights, the
explanation-conditioned regressor predicts an offset, the alpha-gate blends it
onto the proposal, and the refined box is swapped back into the detector's
predictions. Then COCO AP is reported BASELINE vs REFINED for the class and
all-class. This is the missing eval the old run script and rescore.py import.

Faithful to box_regressor.py's contract (imported directly, not reimplemented):
render_channels / BoxRegressor / blend / decode, checkpoint {"model","alpha"},
boxes in the native crop frame lifted to image coords via crop_meta (ox1,oy1).
Only detections that were decomposed (have a .lp) are refined; every other
detection keeps its original box and score, so the number is a fair whole-split
AP, not a matched-subset number.

Two callable surfaces:
  * coco_eval(coco_gt, preds, cat_id=None, tag=None) -> AP/AR dict
    (kept for rescore.py, which does `from eval_coco_ap import coco_eval`)
  * main(): the pipeline eval CLI below.

Usage (SeaClear urchin, refined-box AP on full val):
  python eval_coco_ap.py \
    --rules $ASP --weights $OUT/urchin/alt_animal_urchin_weights.lp \
    --ckpt $OUT/urchin/alt_animal_urchin_regressor.pt \
    --lp-dir $DEC_VAL --emb-dir $DEC_VAL/emb \
    --crop-meta $DEC_VAL/crop_meta.json \
    --preds $PRED_VAL --coco-gt $GT_VAL \
    --class animal_urchin --cat-id 17 \
    --out $OUT/urchin/refined_val.json

Notes:
  * --alpha overrides the checkpoint gate (--alpha 1.0 = full refinement;
    --alpha 0 must reproduce baseline exactly -- a built-in sanity check).
  * needs the embedding dump (--emb-dir): the regressor consumes DINOv3 crop
    features, so decompose the eval split with --dump-emb.

Deps: numpy, pycocotools, torch, cv2, clingo (via learn_weights/box_regressor).
"""
import io, os, json, argparse, contextlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_KEYS = ["AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l",
         "AR1", "AR10", "AR100", "AR_s", "AR_m", "AR_l"]


def _load_gt(coco_gt):
    if isinstance(coco_gt, COCO):
        return coco_gt
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(coco_gt)


def coco_eval(coco_gt, preds, cat_id=None, tag=None):
    """AP/AR dict for a predictions list. None if preds empty. Copies pred dicts
    (loadRes mutates them)."""
    if not preds:
        if tag:
            print(f"[{tag}] no predictions -> skipped")
        return None
    cocoGt = _load_gt(coco_gt)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cocoDt = cocoGt.loadRes([dict(p) for p in preds])
        E = COCOeval(cocoGt, cocoDt, "bbox")
        if cat_id is not None:
            E.params.catIds = [int(cat_id)]
        E.evaluate(); E.accumulate(); E.summarize()
    if tag:
        print(f"[{tag}]" + (f" cat={cat_id}" if cat_id is not None else " all-class"))
        print(buf.getvalue().rstrip("\n"))
    return dict(zip(_KEYS, [float(v) for v in E.stats]))


def _stub_gt(args):
    """RefineSet needs a gt_boxes.json to enumerate samples. For inference we only
    need the stem set with a box; if --gt-boxes is absent, synthesize proposal
    boxes (crop frame) from crop_meta so every decomposed stem loads."""
    if args.gt_boxes:
        return args.gt_boxes
    cmeta = json.load(open(args.crop_meta))
    gt = {stem: [cm[3]-cm[1], cm[4]-cm[2], cm[5]-cm[1], cm[6]-cm[2]]  # image->crop xyxy
          for stem, cm in cmeta.items()}
    path = os.path.join(os.path.dirname(args.crop_meta), "_eval_stub_gt.json")
    json.dump(gt, open(path, "w"))
    return path


def _refine_boxes(args, device):
    """solve -> render -> regressor -> blend for every decomposed target-class
    stem. Returns ({stem: refined_box_crop_xyxy}, alpha)."""
    import torch
    from box_regressor import RefineSet, BoxRegressor, blend, decode

    rs = RefineSet(args.lp_dir, args.emb_dir, _stub_gt(args),
                   args.rules, args.weights, args.cls, S=args.size)
    emb0 = np.load(rs.items[0][4])["emb"]
    net = BoxRegressor(emb_dim=emb0.shape[0]).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state)
    net.eval()
    alpha = args.alpha if args.alpha is not None else (
        ckpt.get("alpha", 1.0) if isinstance(ckpt, dict) else 1.0)

    refined = {}
    with torch.no_grad():
        for i in range(len(rs)):
            stem = rs.items[i][0]
            emb, prim, prop, _gt = rs[i]
            emb = emb.unsqueeze(0).to(device); prim = prim.unsqueeze(0).to(device)
            prop_d = prop.unsqueeze(0).to(device)
            ref = blend(prop_d, decode(prop_d, net(emb, prim)), alpha)[0].cpu().numpy()
            refined[stem] = [float(v) for v in ref]
            if (i + 1) % 500 == 0:
                print(f"    refined {i+1}/{len(rs)}...")
    return refined, float(alpha)


def _swap(preds, cat_id, refined, cmeta):
    """Lift refined crop boxes to image coords via crop_meta origin, swap into
    preds by the (cat,img,proposal-xyxy) box-key. Returns (new_preds, n)."""
    keyed = {}
    for stem, rb in refined.items():
        cm = cmeta.get(stem)
        if cm is None:
            continue
        ox, oy = cm[1], cm[2]
        img_box = [rb[0]+ox, rb[1]+oy, rb[2]+ox, rb[3]+oy]     # crop -> image xyxy
        k = (int(cat_id), int(cm[0]), round(cm[3], 2), round(cm[4], 2),
             round(cm[5], 2), round(cm[6], 2))                 # keyed on PROPOSAL box
        keyed[k] = img_box
    out, n = [], 0
    for d in preds:
        x, y, w, h = d["bbox"]
        k = (d.get("category_id"), d["image_id"], round(x, 2), round(y, 2),
             round(x+w, 2), round(y+h, 2))
        q = dict(d)
        if k in keyed:
            b = keyed[k]
            q["bbox"] = [b[0], b[1], b[2]-b[0], b[3]-b[1]]     # image xyxy -> xywh
            n += 1
        out.append(q)
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--weights", required=True, help="learned weights.lp")
    ap.add_argument("--ckpt", required=True, help="regressor .pt ({'model','alpha'})")
    ap.add_argument("--lp-dir", required=True)
    ap.add_argument("--emb-dir", required=True, help="DINOv3 crop embeddings (.npz per stem)")
    ap.add_argument("--crop-meta", required=True)
    ap.add_argument("--gt-boxes", default=None, help="optional; only for sample enumeration")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--coco-gt", required=True)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--cat-id", type=int, required=True)
    ap.add_argument("--out", required=True, help="write refined predictions json")
    ap.add_argument("--alpha", type=float, default=None, help="override checkpoint alpha-gate")
    ap.add_argument("--size", type=int, default=64, help="render size S (must match training)")
    ap.add_argument("--solve-timeout", type=float, default=0.0,
                    help="cap clingo solve at N sec/stem (0=off). Use if RefineSet hangs on "
                         "a class whose facts make the solve intractable; returns best-so-far.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.solve_timeout and args.solve_timeout > 0:
        try:
            import box_regressor, solve_safe
            tl = float(args.solve_timeout)
            box_regressor.solve = lambda prog: solve_safe.solve_timed(prog, tl)[:2]
            print(f"[eval] clingo solve capped at {tl:.1f}s/stem (solve_safe)")
        except Exception as e:
            print(f"[eval] WARNING: could not enable solve timeout ({e}); running uncapped")

    print(f"[eval] refining {args.cls} detections via {os.path.basename(args.ckpt)} ...")
    refined, alpha = _refine_boxes(args, device)
    print(f"[eval] refined {len(refined)} decomposed dets  (alpha={alpha:.2f})")

    cmeta = json.load(open(args.crop_meta))
    preds = json.load(open(args.preds))
    new_preds, n = _swap(preds, args.cat_id, refined, cmeta)
    json.dump(new_preds, open(args.out, "w"))
    print(f"[eval] swapped {n}/{len(refined)} boxes into preds -> {args.out}")

    gt = _load_gt(args.coco_gt)
    base_c = coco_eval(gt, preds, args.cat_id, None)
    ref_c = coco_eval(gt, new_preds, args.cat_id, None)
    base_a = coco_eval(gt, preds, None, None)
    ref_a = coco_eval(gt, new_preds, None, None)

    MET = ["AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l"]
    print(f"\n== class '{args.cls}' (cat {args.cat_id}):  baseline -> refined  (alpha={alpha:.2f}) ==")
    for k in MET:
        b, r = base_c[k], ref_c[k]
        print(f"   {k:5s}  {b:.4f} -> {r:.4f}  ({r-b:+.4f})")
    print(f"\n== ALL-CLASS (only '{args.cls}' refined) ==")
    for k in MET:
        print(f"   {k:5s}  {base_a[k]:.4f} -> {ref_a[k]:.4f}  ({ref_a[k]-base_a[k]:+.4f})")

    json.dump(dict(cls=args.cls, cat_id=args.cat_id, alpha=alpha, n_refined=n,
                   baseline={k: base_c[k] for k in MET},
                   refined={k: ref_c[k] for k in MET},
                   baseline_all={k: base_a[k] for k in MET},
                   refined_all={k: ref_a[k] for k in MET}),
              open(os.path.splitext(args.out)[0] + "_report.json", "w"), indent=1)
    print(f"\nwrote {os.path.splitext(args.out)[0] + '_report.json'}")


if __name__ == "__main__":
    main()