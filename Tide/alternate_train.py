#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alternate_train.py -- Alternating Explanation Learning (paper Sec. 3.5, eqs 12-13).

Two parameter sets:
  Theta = {w_{C,f}, lambda_{C,g}, L_C, U_C}   explanation weights + cardinality bounds
  Omega                                        explanation-conditioned box regressor

Loop (paper-faithful):
  Theta(0)  <- the manually specified weights/bounds inside the rules .lp
               (fallback: all-ones if the class has none)
  repeat t = 0..iters-1:
    [eq 12]  generate explanations with Theta(t) for all training detections,
             train Omega(t+1) on them (jitter + GIoU), pick alpha-gate on the
             internal validation slice
    [eq 13]  freeze Omega(t+1); Optuna over Theta, where each trial's loss is
             mean(1 - IoU(R_Omega(B, X(Theta)), B_gt)) THROUGH the frozen
             regressor on a fixed training subsample (NOT the union-box
             surrogate of learn_weights.py)
    stop when the internal-validation IoU stops improving (--tol)

Saves the best (Theta*, Omega*, alpha) pairing seen across iterations:
  <out-prefix>_weights.lp     drop-in for box_regressor / eval_coco_ap / viz
  <out-prefix>_regressor.pt   dict checkpoint {"model", "alpha", ...}
  <out-prefix>_history.json

Must sit next to box_regressor.py (patched) and learn_weights.py.

Example:
  python alternate_train.py --rules ../object_explanation_v2.lp \
      --lp-dir $OUT/urchin/train_disjoint/lp --emb-dir $OUT/urchin/train_disjoint/emb \
      --gt $OUT/urchin/train_disjoint/gt_boxes.json --class animal_urchin \
      --iters 3 --trials 60 --epochs 15 --out-prefix $OUT/urchin/alt_animal_urchin
"""
import os, re, json, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import optuna
from learn_weights import (load_rules, solve, parse_explains, discover_features,
                           PENALTY_NAMES)
from box_regressor import (RefineSet, JitterView, BoxRegressor, decode, blend,
                           giou_loss, split_indices, evaluate, report)


# ----------------------------------------------------------------- Theta helpers
def weight_block(cls, wmap, L, U):
    lines = [f"bound({cls},semantic_region,{L},{U})."]
    lines += [f"weight({cls},{f},{w})." for f, w in sorted(wmap.items())]
    return "\n".join(lines) + "\n"


def theta0_from_rules(rules_path, cls, feats):
    """Paper: Theta(0) is manually specified -- read the example weight()/bound()
    facts for this class straight from the rules file. Features present in the
    search space but absent from the file default to 1 (0 for penalties is too
    inert a start). Returns (wmap, L, U, n_found)."""
    txt = open(rules_path).read()
    wmap = {}
    for m in re.finditer(rf"weight\({re.escape(cls)},(.+?),(-?\d+)\)\.", txt):
        wmap[m.group(1)] = int(m.group(2))
    mb = re.search(rf"bound\({re.escape(cls)},semantic_region,(\d+),(\d+)\)\.", txt)
    L, U = (int(mb.group(1)), int(mb.group(2))) if mb else (1, 2)
    n_found = len(wmap)
    for f in feats:
        wmap.setdefault(f, 1)
    wmap = {f: w for f, w in wmap.items() if f in feats}   # keep only live features
    return wmap, L, U, n_found


class MutableRefineSet(RefineSet):
    """RefineSet whose weight block can be swapped; swapping invalidates the
    per-stem explanation cache so the next pass re-solves with new Theta."""
    def set_weight_block(self, wb):
        self.wb = wb
        self._cache = {}


# ----------------------------------------------------------------- Omega training
def train_regressor(ds, tr_idx, va_idx, args, dev):
    """eq 12: explanations fixed (ds.wb fixed), fit Omega. Returns (net, alpha, va_metrics)."""
    if args.jitter_shift > 0:
        tr_view = JitterView(ds, tr_idx, args.jitter_shift, args.jitter_scale, args.jitter_repeat)
    else:
        tr_view = Subset(ds, tr_idx)
    tr_dl = DataLoader(tr_view, batch_size=args.batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(Subset(ds, va_idx), batch_size=args.batch, shuffle=False, num_workers=0)

    emb_dim = ds[0][0].shape[0]
    net = BoxRegressor(emb_dim).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for ep in range(args.epochs):
        net.train(); tot = nb = 0
        for emb, prim, prop, gt in tr_dl:
            emb, prim = emb.to(dev), prim.to(dev)
            prop, gt = prop.to(dev), gt.to(dev)
            loss, _tr_iou = giou_loss(decode(prop, net(emb, prim)), gt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if ep == 0 or (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            print(f"      epoch {ep+1:>3}/{args.epochs}  train GIoU loss {tot/max(1,nb):.4f}")

    # alpha-gate sweep on the internal validation slice (ties -> smaller alpha)
    cand = [float(a) for a in args.alphas.split(",")]
    sweep = [(evaluate(net, va_dl, dev, alpha=a)["ref"], -a) for a in cand]
    alpha = -max(sweep)[1]
    va = evaluate(net, va_dl, dev, alpha=alpha)
    print(f"      alpha sweep: " + "  ".join(f"{a:.2f}:{r:.3f}" for (r, _), a in zip(sweep, cand))
          + f"  -> alpha={alpha:.2f}   val IoU {va['det']:.3f} -> {va['ref']:.3f}")
    return net, alpha, va


# ----------------------------------------------------------------- Theta search (eq 13)
def optimize_theta(ds, net, alpha, sub_idx, feats, wmap0, L0, U0, args, dev, seed):
    """Freeze Omega; search Theta by loss THROUGH the regressor on a fixed subsample."""
    sub_dl = DataLoader(Subset(ds, sub_idx), batch_size=args.batch, shuffle=False, num_workers=0)

    def make_wb(trial):
        wmap = {}
        for f in feats:
            lo = 0 if f in PENALTY_NAMES else args.wmin
            wmap[f] = trial.suggest_int(f"w_{f}", lo, args.wmax)
        L = trial.suggest_int("bound_L", 1, 2)
        U = L + trial.suggest_int("bound_U_extra", 0, 3)
        return wmap, L, U

    def objective(trial):
        wmap, L, U = make_wb(trial)
        ds.set_weight_block(weight_block(args.cls, wmap, L, U))
        m = evaluate(net, sub_dl, dev, alpha=alpha)     # re-solves under new Theta
        return 1.0 - m["ref"]                           # eq 13 / eq 10 through R_Omega

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    # seed the search with the incumbent Theta so it can only improve on it
    enq = {f"w_{f}": wmap0[f] for f in feats}
    enq["bound_L"] = max(1, min(2, L0)); enq["bound_U_extra"] = max(0, min(3, U0 - enq["bound_L"]))
    study.enqueue_trial(enq)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_params
    wmap = {f: best[f"w_{f}"] for f in feats}
    L = best["bound_L"]; U = L + best["bound_U_extra"]
    return wmap, L, U, 1.0 - study.best_value


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--lp-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--trials", type=int, default=60, help="Optuna trials per outer iteration")
    ap.add_argument("--trial-sample", type=int, default=120,
                    help="training detections evaluated per Theta trial (cost control)")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="stop when val IoU improves less than this over an iteration")
    # regressor / eq-12 knobs (mirror box_regressor.py)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--split-by", choices=["detection", "image"], default="image")
    ap.add_argument("--jitter-shift", type=float, default=0.15)
    ap.add_argument("--jitter-scale", type=float, default=0.25)
    ap.add_argument("--jitter-repeat", type=int, default=4)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    # Theta search space
    ap.add_argument("--wmin", type=int, default=-5)
    ap.add_argument("--wmax", type=int, default=8)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # dataset with a dummy weight block first (need instances to discover features)
    tmp_wb = os.path.join(os.path.dirname(args.out_prefix) or ".", f"_wb_{args.cls}_tmp.lp")
    open(tmp_wb, "w").write(f"bound({args.cls},semantic_region,1,2).\n")
    ds = MutableRefineSet(args.lp_dir, args.emb_dir, args.gt, args.rules, tmp_wb, args.cls, args.S)
    print(f"[{args.cls}] {len(ds)} training detections   device={dev}")

    rules = load_rules(args.rules)
    feats = discover_features(rules, [it[1] for it in ds.items[:min(20, len(ds.items))]])
    print(f"Theta space: {len(feats)} feature weights + bounds (L,U)")

    wmap, L, U, n0 = theta0_from_rules(args.rules, args.cls, feats)
    print(f"Theta(0): {n0} weights read from rules file for '{args.cls}'"
          + ("" if n0 else "  (none found -> all-ones init)"))

    tr_idx, va_idx = split_indices(ds.items, args.val_frac, args.split_by, args.seed)
    rng = np.random.RandomState(args.seed)
    sub_idx = list(rng.permutation(tr_idx)[:min(args.trial_sample, len(tr_idx))])
    print(f"split: train {len(tr_idx)} / internal-val {len(va_idx)}   "
          f"theta-trial subsample {len(sub_idx)}")

    hist, best = [], {"val_iou": -1.0}
    for t in range(args.iters):
        print(f"\n================ iteration {t}  (eq 12: train Omega) ================")
        wb = weight_block(args.cls, wmap, L, U)
        ds.set_weight_block(wb)
        t0 = time.time()
        net, alpha, va = train_regressor(ds, tr_idx, va_idx, args, dev)
        report(f"it{t}-va", va)

        if va["ref"] > best["val_iou"] + 0.0:      # keep the best pairing seen
            best = dict(val_iou=va["ref"], wb=wb, alpha=alpha, iter=t,
                        state={k: v.cpu() for k, v in net.state_dict().items()})

        print(f"---------------- iteration {t}  (eq 13: optimize Theta) ----------------")
        wmap, L, U, sub_iou = optimize_theta(ds, net, alpha, sub_idx, feats,
                                             wmap, L, U, args, dev, seed=args.seed + t)
        # measure the new Theta on the val slice through the SAME frozen Omega
        ds.set_weight_block(weight_block(args.cls, wmap, L, U))
        va2 = evaluate(net, DataLoader(Subset(ds, va_idx), batch_size=args.batch,
                                       shuffle=False, num_workers=0), dev, alpha=alpha)
        gain = va2["ref"] - va["ref"]
        print(f"Theta(t+1): subsample IoU {sub_iou:.3f} | val IoU {va['ref']:.3f} -> "
              f"{va2['ref']:.3f} ({gain:+.4f})   elapsed {time.time()-t0:.0f}s")
        hist.append(dict(iter=t, val_iou_after_omega=float(va["ref"]),
                         val_iou_after_theta=float(va2["ref"]), alpha=float(alpha),
                         sub_iou=float(sub_iou), L=int(L), U=int(U)))
        if va2["ref"] > best["val_iou"]:
            best = dict(val_iou=va2["ref"], wb=weight_block(args.cls, wmap, L, U),
                        alpha=alpha, iter=t,
                        state={k: v.cpu() for k, v in net.state_dict().items()})

        if t > 0 and gain < args.tol and va2["ref"] <= hist[t-1]["val_iou_after_theta"] + args.tol:
            print(f"converged (gain {gain:+.4f} < tol {args.tol}) -- stopping early")
            break

    # ---- save the best (Theta*, Omega*, alpha) pairing ----
    w_out = f"{args.out_prefix}_weights.lp"
    with open(w_out, "w") as fh:
        fh.write(f"% alternating optimization (paper eq 12-13); best val IoU "
                 f"{best['val_iou']:.4f} at iteration {best['iter']}\n" + best["wb"])
    r_out = f"{args.out_prefix}_regressor.pt"
    torch.save({"model": best["state"], "alpha": best["alpha"],
                "emb_dim": ds[0][0].shape[0], "cls": args.cls}, r_out)
    json.dump(hist, open(f"{args.out_prefix}_history.json", "w"), indent=1)
    os.remove(tmp_wb)
    print(f"\nBEST: val IoU {best['val_iou']:.4f} (iteration {best['iter']}, "
          f"alpha={best['alpha']:.2f})")
    print(f"wrote {w_out}\nwrote {r_out}\nwrote {args.out_prefix}_history.json")


if __name__ == "__main__":
    main()