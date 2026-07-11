#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_selector.py -- gated primitive-box selector.

Trains a small regressor to predict each candidate box's IoU-to-GT from
features (build_candidates.py rows), then at inference picks, per detection, the
best-PREDICTED candidate -- but swaps it in ONLY if it beats the predicted
proposal quality by a margin. The margin gate is the whole game: the localize
probe showed 82/253 loose dets have no candidate better than the proposal and
ellipse-swap-always is -0.106 AP, so an ungated selector goes negative. We sweep
the margin and report AP at each, plus the always-swap and oracle bounds.

Why predict IoU and gate on it, rather than classify "swap/keep": the proposal
is itself a candidate with a predicted quality, so the model learns a calibrated
"is any box better than doing nothing" decision, and the margin trades recall
(swaps that help) against precision (avoiding swaps that hurt).

Trains on the TRAIN candidate npz, evaluates on VAL: predicts, gates, swaps the
chosen box into preds by the build_candidates box (image xywh) via crop-meta key,
runs COCO AP baseline vs selected. Reports per-margin and picks the margin that
maximizes val AP75 (the metric the loose band drives).

Model: gradient-boosted-ish? No -- keep deps minimal: standardized features +
a small torch MLP (or ridge fallback if torch absent). Target is IoU in [0,1],
sigmoid output, MSE.

Usage:
  python train_selector.py \
    --train-cand $OUT/urchin/cand_train.npz \
    --val-cand   $OUT/urchin/cand_val.npz \
    --val-crop-meta $DEC_VAL/crop_meta.json \
    --preds $PRED_VAL --coco-gt $GT_VAL --class animal_urchin --cat-id 17 \
    --out $OUT/urchin/selected_val.json

Deps: numpy, pycocotools; torch (MLP) or falls back to numpy ridge.
"""
import os, io, json, argparse, contextlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MET = ["AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l"]


def coco_eval(cocoGt, preds, cat_id):
    if not preds:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        dt = cocoGt.loadRes([dict(p) for p in preds])
        E = COCOeval(cocoGt, dt, "bbox"); E.params.catIds = [int(cat_id)]
        E.evaluate(); E.accumulate(); E.summarize()
    s = E.stats
    return dict(AP=s[0], AP50=s[1], AP75=s[2], AP_s=s[3], AP_m=s[4], AP_l=s[5])


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum()*(rb**2).sum())
    return float((ra*rb).sum()/d) if d > 0 else float("nan")


def fit_predict(Xtr, ytr, Xva, seed=0, epochs=300, lr=0.05, wd=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n, Xva_n = (Xtr-mu)/sd, (Xva-mu)/sd
    try:
        import torch
        torch.manual_seed(seed)
        Xt = torch.tensor(Xtr_n); yt = torch.tensor(ytr)
        net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.ReLU(),
                                  torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            p = torch.sigmoid(net(Xt).squeeze(1))
            loss = ((p - yt)**2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(Xva_n)).squeeze(1)).numpy()
    except Exception:
        # ridge fallback (closed form) on logit-ish target
        lam = 1.0
        A = Xtr_n.T @ Xtr_n + lam*np.eye(Xtr_n.shape[1])
        w = np.linalg.solve(A, Xtr_n.T @ ytr)
        return np.clip(Xva_n @ w, 0, 1)


def choose(pred, y, box_img, stems, cand, prop_iou, margin):
    """Per detection: pick best-predicted candidate; swap only if its predicted
    quality beats the predicted PROPOSAL quality by >= margin. Returns
    {stem: chosen_box_img_xywh} for swapped dets, and stats."""
    by = {}
    for i, s in enumerate(stems):
        by.setdefault(s, []).append(i)
    swaps, n_swap, realized_gain, wrong = {}, 0, [], 0
    for s, idx in by.items():
        pi = [i for i in idx if cand[i] == "proposal"][0]
        pred_prop = pred[pi]
        best_i = max(idx, key=lambda i: pred[i])
        if cand[best_i] == "proposal":
            continue
        if pred[best_i] - pred_prop >= margin:
            swaps[s] = box_img[best_i]
            n_swap += 1
            realized_gain.append(y[best_i] - y[pi])   # true IoU change from this swap
            if y[best_i] < y[pi] - 1e-6:
                wrong += 1
    return swaps, dict(n_swap=n_swap, wrong=wrong,
                       mean_true_gain=float(np.mean(realized_gain)) if realized_gain else 0.0,
                       sum_true_gain=float(np.sum(realized_gain)) if realized_gain else 0.0)


def swap_into(preds, cat_id, swaps, cmeta):
    """swaps keyed by stem -> box_img xywh. Join to preds via proposal box-key."""
    keyed = {}
    for stem, bx in swaps.items():
        cm = cmeta.get(stem)
        if cm is None:
            continue
        k = (int(cat_id), int(cm[0]), round(cm[3], 2), round(cm[4], 2),
             round(cm[5], 2), round(cm[6], 2))
        keyed[k] = [float(v) for v in bx]
    out, n = [], 0
    for d in preds:
        x, y, w, h = d["bbox"]
        k = (d.get("category_id"), d["image_id"], round(x, 2), round(y, 2),
             round(x+w, 2), round(y+h, 2))
        q = dict(d)
        if k in keyed:
            q["bbox"] = keyed[k]; n += 1
        out.append(q)
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cand", required=True)
    ap.add_argument("--val-cand", required=True)
    ap.add_argument("--val-crop-meta", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--coco-gt", required=True)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--cat-id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margins", default="0.0,0.02,0.05,0.08,0.12,0.2")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    tr = np.load(args.train_cand, allow_pickle=True)
    va = np.load(args.val_cand, allow_pickle=True)
    Xtr, ytr = tr["X"], tr["y"]
    Xva, yva = va["X"], va["y"]
    stems_va, cand_va = list(va["stem"]), list(va["cand"])
    box_va, prop_iou_va = va["box_img"], va["prop_iou"]
    margins = [float(m) for m in args.margins.split(",")]

    with contextlib.redirect_stdout(io.StringIO()):
        cocoGt = COCO(args.coco_gt)
    preds = json.load(open(args.preds))
    cmeta = json.load(open(args.val_crop_meta))
    base = coco_eval(cocoGt, preds, args.cat_id)

    # multi-seed prediction on val
    preds_va = np.zeros((args.seeds, len(yva)), np.float32)
    for s in range(args.seeds):
        preds_va[s] = fit_predict(Xtr, ytr, Xva, seed=s)
    pred_mean = preds_va.mean(0)

    # candidate-level regression quality
    print(f"== selector prediction quality (val, {len(yva)} candidate rows) ==")
    print(f"   Spearman(pred, true IoU): {spearman(pred_mean, yva):+.3f}  "
          f"(seed std {np.std([spearman(preds_va[s], yva) for s in range(args.seeds)]):.3f})")

    # oracle + always-swap bounds
    orc_swaps = {}
    by = {}
    for i, s in enumerate(stems_va):
        by.setdefault(s, []).append(i)
    for s, idx in by.items():
        pi = [i for i in idx if cand_va[i] == "proposal"][0]
        bi = max(idx, key=lambda i: yva[i])                 # GT-oracle best
        if yva[bi] > yva[pi] + 1e-6:
            orc_swaps[s] = box_va[bi]
    orc_preds, _ = swap_into(preds, args.cat_id, orc_swaps, cmeta)
    orc = coco_eval(cocoGt, orc_preds, args.cat_id)

    print(f"\n== bounds ==")
    print(f"   {'variant':22s} {'AP':>8} {'AP75':>8}  n_swap")
    print(f"   {'baseline':22s} {base['AP']:8.4f} {base['AP75']:8.4f}   0")
    print(f"   {'oracle-select':22s} {orc['AP']:8.4f} {orc['AP75']:8.4f}   {len(orc_swaps)}")

    # margin sweep
    print(f"\n== margin sweep (selector) ==")
    print(f"   {'margin':>6} {'AP':>8} {'AP75':>8}  {'n_swap':>6} {'wrong':>6}  true_ΣIoU")
    best = None
    for m in margins:
        swaps, st = choose(pred_mean, yva, box_va, stems_va, cand_va, prop_iou_va, m)
        new_preds, n = swap_into(preds, args.cat_id, swaps, cmeta)
        r = coco_eval(cocoGt, new_preds, args.cat_id)
        tag = ""
        if best is None or r["AP75"] > best[1]["AP75"]:
            best = (m, r, new_preds, st); tag = "  <- best AP75"
        print(f"   {m:6.2f} {r['AP']:8.4f} {r['AP75']:8.4f}  {st['n_swap']:6d} "
              f"{st['wrong']:6d}  {st['sum_true_gain']:+8.3f}{tag}")

    m_best, r_best, preds_best, st_best = best
    json.dump(preds_best, open(args.out, "w"))
    print(f"\n== chosen selector: margin={m_best:.2f} -> {args.out} ==")
    for k in MET:
        print(f"   {k:5s}  {base[k]:.4f} -> {r_best[k]:.4f}  ({r_best[k]-base[k]:+.4f})   "
              f"oracle {orc[k]:.4f}")
    # all-class effect
    ra = coco_eval(cocoGt, preds_best, None) if False else None
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(cocoGt, cocoGt.loadRes([dict(p) for p in preds_best]), "bbox")
        E.evaluate(); E.accumulate(); E.summarize()
    Eb = COCOeval(cocoGt, cocoGt.loadRes([dict(p) for p in preds]), "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        Eb.evaluate(); Eb.accumulate(); Eb.summarize()
    print(f"\n== ALL-CLASS (only '{args.cls}' selected) ==")
    for i, k in enumerate(MET):
        print(f"   {k:5s}  {Eb.stats[i]:.4f} -> {E.stats[i]:.4f}  ({E.stats[i]-Eb.stats[i]:+.4f})")

    json.dump(dict(cls=args.cls, cat_id=args.cat_id, margin=m_best,
                   n_swap=st_best["n_swap"], n_wrong=st_best["wrong"],
                   spearman=spearman(pred_mean, yva),
                   selected={k: r_best[k] for k in MET},
                   baseline={k: base[k] for k in MET},
                   oracle={k: orc[k] for k in MET}),
              open(os.path.splitext(args.out)[0] + "_report.json", "w"), indent=1)
    print(f"\nwrote {os.path.splitext(args.out)[0] + '_report.json'}")


if __name__ == "__main__":
    main()