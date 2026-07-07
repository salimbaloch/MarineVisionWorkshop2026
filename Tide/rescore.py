#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rescore.py -- explanation-based RE-SCORING (the ranking channel of AP).

Hypothesis: a true detection at low detector score still admits a COHERENT
explanation (body + consistent boundary at low ASP cost); background clutter
does not. If explanation coherence separates matched from unmatched detections,
re-ranking lifts AP without moving a single box.

Pipeline:
  1. per detection: solve the ASP once -> feature vector
       [det score, candidate/selected counts, support/edge-conf stats of the
        selection, ASP optimization cost (raw + per-atom), union-box agreement
        with the proposal, coverage, size/aspect]
     label = stem present in gt_boxes.json (matched at decompose gt-iou)
  2. DIAGNOSTIC: single-feature AUCs (does the signal exist at all?)
  3. tiny logistic regression (torch, no sklearn) -> calibrated p = P(matched)
  4. blend: new_score = (1-gamma)*det_score + gamma*p ; gamma swept on a
     held-out slice by binary average precision
  5. swap re-scored detections into the baseline predictions json (boxes
     untouched, non-decomposed detections untouched) -> COCO AP baseline vs
     re-scored, overall + per class.

Feature extraction is cached (--cache-*) because ~1 solve/detection dominates.

Run from $TIDE (imports learn_weights + eval_coco_ap).

Example:
  python rescore.py --rules $ASP --weights $OUT/urchin/alt_animal_urchin_weights.lp \
    --train-lp-dir $DISJ/lp --train-crop-meta $DISJ/crop_meta.json \
    --train-labels $DISJ/gt_boxes.json \
    --val-lp-dir $DEC_VAL --val-crop-meta $DEC_VAL/crop_meta.json \
    --val-labels $DEC_VAL/gt_boxes.json \
    --preds $PRED_VAL --coco-gt $GT_VAL --class animal_urchin --cat-id 17 \
    --cache-train $OUT/urchin/feat_train.npz --cache-val $OUT/urchin/feat_val.npz \
    --out $OUT/urchin/rescored_animal_urchin.json
"""
import os, re, json, glob, argparse
import numpy as np
import torch

from learn_weights import load_rules, solve, parse_facts, parse_explains

FEAT_NAMES = ["det_score", "log_n_cand", "n_sel", "frac_sel", "body_sel",
              "n_boundary_sel", "mean_supp_sel", "max_supp_sel",
              "mean_edgeconf_bnd", "mean_conf_sel", "cost0", "cost_per_sel",
              "union_iou", "coverage", "log_area", "aspect"]

_SUPP = re.compile(r"semantic_support\((\w+),(-?\d+)\)")
_EDGE = re.compile(r"edge_confidence\((\w+),(-?\d+)\)")
_CONF = re.compile(r"confidence\((\w+),(-?\d+)\)")


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def _cost_scalar(cost):
    if cost is None:
        return 0.0
    if isinstance(cost, (list, tuple)):
        return float(cost[0]) if len(cost) else 0.0
    return float(cost)


def features_one(facts, meta, det_score, rules, wb):
    try:
        atoms, cost = solve(rules + "\n" + facts + "\n" + wb)
        sel = parse_explains(atoms)
    except Exception:
        sel, cost = set(), 0.0
    supp = {m.group(1): int(m.group(2)) for m in _SUPP.finditer(facts)}
    econ = {m.group(1): int(m.group(2)) for m in _EDGE.finditer(facts)}
    conf = {m.group(1): int(m.group(2)) for m in _CONF.finditer(facts)}
    boxes, bids, body = meta["boxes"], meta["boundary_ids"], meta["body"]
    prop = meta["proposal"]

    cand = set(boxes) | set(supp) | set(conf)
    n_sel = len(sel)
    bnd_sel = [p for p in sel if p in bids]
    supp_sel = [supp[p] for p in sel if p in supp]
    conf_sel = [conf[p] for p in sel if p in conf]
    econ_bnd = [econ[p] for p in bnd_sel if p in econ]

    sb = [boxes[p] for p in sel if p in boxes]
    if sb:
        ub = (min(b[0] for b in sb), min(b[1] for b in sb),
              max(b[2] for b in sb), max(b[3] for b in sb))
        union_iou = _iou(ub, prop)
        pa = max(1.0, (prop[2]-prop[0]) * (prop[3]-prop[1]))
        coverage = min(2.0, (ub[2]-ub[0]) * (ub[3]-ub[1]) / pa)
    else:
        union_iou, coverage = 0.0, 0.0
    pw, ph = max(1.0, prop[2]-prop[0]), max(1.0, prop[3]-prop[1])
    c0 = _cost_scalar(cost)
    return [det_score, float(np.log1p(len(cand))), float(n_sel),
            n_sel / max(1, len(cand)), float(body in sel), float(len(bnd_sel)),
            float(np.mean(supp_sel)) if supp_sel else 0.0,
            float(np.max(supp_sel)) if supp_sel else 0.0,
            float(np.mean(econ_bnd)) if econ_bnd else 0.0,
            float(np.mean(conf_sel)) if conf_sel else 0.0,
            c0, c0 / max(1, n_sel), union_iou, coverage,
            float(np.log1p(pw * ph)), float(np.clip(pw / ph, 0.1, 10.0))]


def extract(lp_dir, crop_meta_path, labels_path, rules, wb, cls, cache=None, limit=0):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"  [cache] {cache}: {len(z['y'])} rows")
        return z["X"], z["y"], list(z["stems"])
    cmeta = json.load(open(crop_meta_path))
    labels = set(json.load(open(labels_path))) if labels_path else set()
    paths = sorted(glob.glob(os.path.join(lp_dir, "*.lp")))
    if limit:
        paths = paths[:limit]
    X, y, stems = [], [], []
    skipped = 0
    for i, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in cmeta:
            skipped += 1
            continue
        facts = open(p).read()
        meta = parse_facts(facts)
        if meta.get("cls") != cls:
            skipped += 1
            continue
        X.append(features_one(facts, meta, float(cmeta[stem][7]), rules, wb))
        y.append(1.0 if stem in labels else 0.0)
        stems.append(stem)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(paths)} solved...")
    X = np.array(X, np.float32); y = np.array(y, np.float32)
    print(f"  extracted {len(y)} rows ({int(y.sum())} matched / {int((1-y).sum())} unmatched, "
          f"{skipped} skipped)")
    if cache:
        np.savez_compressed(cache, X=X, y=y, stems=np.array(stems))
        print(f"  cached -> {cache}")
    return X, y, stems


# --------------------------------------------------------------- metrics (no sklearn)
def auc(s, y):
    s = np.asarray(s, np.float64); y = np.asarray(y, np.float64)
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    np_, nn = y.sum(), (1 - y).sum()
    if np_ == 0 or nn == 0:
        return float("nan")
    return float((r[y == 1].sum() - np_ * (np_ + 1) / 2) / (np_ * nn))


def avg_precision(s, y):
    o = np.argsort(-np.asarray(s)); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(1.0, y.sum()))


# --------------------------------------------------------------- logistic (torch)
def fit_logistic(Xtr, ytr, epochs=600, lr=0.05, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = torch.tensor((Xtr - mu) / sd)
    yt = torch.tensor(ytr)
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos_w = torch.tensor([(1 - ytr).sum() / max(1.0, float(ytr.sum()))])
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(Xn @ w + b.expand(len(yt)), yt)
        loss.backward(); opt.step()

    def predict(X):
        Xn = torch.tensor((X - mu) / sd)
        with torch.no_grad():
            return torch.sigmoid(Xn @ w + b).numpy()
    coef = dict(zip(FEAT_NAMES, (w.detach().numpy() / sd).round(4).tolist()))
    return predict, coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--train-lp-dir", required=True)
    ap.add_argument("--train-crop-meta", required=True)
    ap.add_argument("--train-labels", required=True, help="gt_boxes.json (presence = matched)")
    ap.add_argument("--val-lp-dir", required=True)
    ap.add_argument("--val-crop-meta", required=True)
    ap.add_argument("--val-labels", default=None, help="optional: val gt_boxes.json for AUC report")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--coco-gt", required=True)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--cat-id", type=int, default=None)
    ap.add_argument("--out", required=True, help="re-scored predictions json")
    ap.add_argument("--cache-train", default=None)
    ap.add_argument("--cache-val", default=None)
    ap.add_argument("--gammas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--val-frac", type=float, default=0.2, help="train slice held out for gamma")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap detections per split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rules = load_rules(args.rules)
    wb = open(args.weights).read()

    print("== extracting TRAIN features ==")
    Xtr, ytr, _ = extract(args.train_lp_dir, args.train_crop_meta, args.train_labels,
                          rules, wb, args.cls, args.cache_train, args.limit)
    print("== extracting VAL features ==")
    Xva, yva, stems_va = extract(args.val_lp_dir, args.val_crop_meta, args.val_labels,
                                 rules, wb, args.cls, args.cache_val, args.limit)

    # ---- diagnostic: single-feature AUCs on train ----
    print("\n== single-feature AUC (train, matched vs unmatched) ==")
    for j, name in enumerate(FEAT_NAMES):
        a = auc(Xtr[:, j], ytr)
        a = max(a, 1 - a)                      # direction-agnostic
        flag = "  <-- signal" if a > 0.60 and name != "det_score" else ""
        print(f"    {name:18s} {a:.3f}{flag}")
    print("    (det_score is the baseline ranker; explanation features above "
          "~0.60 carry signal)")

    # ---- fit on train minus a gamma-selection slice ----
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(ytr))
    n_hold = max(1, int(len(idx) * args.val_frac))
    hold, fit = idx[:n_hold], idx[n_hold:]
    predict, coef = fit_logistic(Xtr[fit], ytr[fit], seed=args.seed)
    p_hold, p_va = predict(Xtr[hold]), predict(Xva)

    print("\n== learned coefficients (standardized) ==")
    for k, v in sorted(coef.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:18s} {v:+.4f}")

    # ---- gamma sweep on the held-out train slice ----
    s_hold = Xtr[hold, 0]
    print("\n== gamma sweep (held-out train slice, binary AP) ==")
    base_ap = avg_precision(s_hold, ytr[hold])
    best_g, best_ap = 0.0, base_ap
    for g in [float(x) for x in args.gammas.split(",")]:
        apv = avg_precision((1 - g) * s_hold + g * p_hold, ytr[hold])
        mark = ""
        if apv > best_ap:
            best_g, best_ap, mark = g, apv, "  <-- best"
        print(f"    gamma={g:.2f}  AP={apv:.4f}{mark}")
    print(f"    baseline (gamma=0): {base_ap:.4f}   chosen gamma={best_g:.2f}")
    if args.val_labels:
        sb = (1 - best_g) * Xva[:, 0] + best_g * p_va
        print(f"\n  val AUC: det_score {max(auc(Xva[:, 0], yva), 1-auc(Xva[:, 0], yva)):.3f}  "
              f"blended {max(auc(sb, yva), 1-auc(sb, yva)):.3f}")

    # ---- apply to the baseline predictions ----
    cmeta = json.load(open(args.val_crop_meta))
    blended = (1 - best_g) * Xva[:, 0] + best_g * p_va
    new_score = {}
    for stem, p in zip(stems_va, blended):
        cm = cmeta[stem]
        key = (
            args.cat_id,
            int(cm[0]),
            round(cm[3], 2),
            round(cm[4], 2),
            round(cm[5], 2),
            round(cm[6], 2),
        )
        new_score[key] = float(p)

    preds = json.load(open(args.preds))
    out, n_swap = [], 0

    for d in preds:
        x, y_, w, h = d["bbox"]
        key = (
            d.get("category_id"),
            d["image_id"],
            round(x, 2),
            round(y_, 2),
            round(x + w, 2),
            round(y_ + h, 2),
        )

        q = dict(d)
        if key in new_score:
            q["score"] = new_score[key]
            n_swap += 1

        out.append(q)
    json.dump(out, open(args.out, "w"))
    print(f"\nre-scored {n_swap}/{len(new_score)} decomposed detections inside "
          f"{len(preds)} predictions -> {args.out}")

    # ---- COCO AP: baseline vs re-scored ----
    from eval_coco_ap import coco_eval
    b_all = coco_eval(args.coco_gt, preds, None, "BASELINE")
    r_all = coco_eval(args.coco_gt, out, None, "RE-SCORED")
    pairs = [("ALL-CLASS", b_all, r_all)]
    if args.cat_id is not None:
        b_c = coco_eval(args.coco_gt, preds, args.cat_id, "BASELINE")
        r_c = coco_eval(args.coco_gt, out, args.cat_id, "RE-SCORED")
        pairs.append((f"CLASS {args.cat_id}", b_c, r_c))
    for tag, b, r in pairs:
        if not b or not r:
            continue
        print(f"\n{tag} delta (re-scored - baseline):")
        for k in ("AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l"):
            print(f"    {k:5s}  {b[k]:.4f} -> {r[k]:.4f}   ({r[k]-b[k]:+.4f})")


if __name__ == "__main__":
    main()