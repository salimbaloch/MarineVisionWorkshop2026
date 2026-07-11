#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faithfulness.py -- interpretability metrics for the symbolic explanations.
CPU only. Reads decomposition .lp facts + make_labels labels (+ optional
gt_boxes for grounding). No ASP solve needed for the core metrics -- the atom
attributes (objectness/coherence/support/fit_error) are already facts, so this
is fast and runs on every class x dataset.

Reports three NAMED, defensible properties (deliberately not umbrella'd as
"faithfulness", which has a stricter XAI meaning this post-hoc pipeline can't
claim):

  GROUNDING     the symbolic body atom lands on the true object.
                mean IoU(support-region bbox, GT) and IoU(ellipse, GT) over
                matched (TP) detections. High => the explanation describes the
                real object, not a hallucination.

  DISCRIMINATIVENESS
                explanation coherence tracks object presence.
                AUC(feature, matched50) and TP-vs-FP mean gap for each of:
                body objectness / coherence / max support / fit_error /
                mean edge-confidence / mean confidence. High AUC => the
                explanation carries real signal about whether a detection is a
                true object, independent of being redundant with det_score for
                RANKING.

  SELECTIVITY   the explanation distinguishes clean objects from clutter.
                foreign-object count and boundary count on TP vs FP (from facts,
                no solve); optionally, with --solve, body-selection rate and
                #selected atoms on TP vs FP.

Emits <out> report json (per-class) consumed by compile_table.py.

Usage:
  python faithfulness.py --lp-dir $DEC_VAL \
    --labels $OUT/urchin/labels_val.json \
    --gt-boxes $DEC_VAL/gt_boxes.json \
    --class animal_urchin --dataset seaclear \
    --out $OUT/urchin/faith_val.json

Deps: numpy. (--solve adds clingo + learn_weights.)
"""
import os, re, glob, json, argparse
import numpy as np

_PROP = re.compile(r"proposal_box\(\w+,box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\)")
_PRIM = re.compile(r"primitive_box\((\w+),box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\)")
_RGN = re.compile(r"region_box\((\w+),box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\)")
_OBJN = re.compile(r"region_objectness\((\w+),(-?\d+)\)")
_COHR = re.compile(r"region_coherence\((\w+),(-?\d+)\)")
_SUPP = re.compile(r"semantic_support\((\w+),(-?\d+)\)")
_FITE = re.compile(r"fit_error\((\w+),(-?\d+)\)")
_ECON = re.compile(r"edge_confidence\((\w+),(-?\d+)\)")
_CONF = re.compile(r"confidence\((\w+),(-?\d+)\)")
_ROLE = re.compile(r"region_role\((\w+),(\w+)\)")
_BND = re.compile(r"boundary_primitive\((\w+)\)")

FEATS = ["body_objness", "body_coher", "max_support", "neg_fit_error",
         "mean_edgeconf", "mean_conf"]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def auc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, float)
    m = ~np.isnan(score)
    score, y = score[m], y[m]
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0 or len(y) == 0:
        return float("nan")
    order = np.argsort(score); r = np.empty(len(score)); r[order] = np.arange(1, len(score)+1)
    a = (r[y == 1].sum() - npos*(npos+1)/2) / (npos*nneg)
    return float(max(a, 1 - a))            # direction-agnostic


def parse(path, ellipse_id):
    txt = open(path).read()
    mp = _PROP.search(txt)
    if not mp:
        return None
    prop = [int(v) for v in mp.groups()]
    objn = {a: int(b) for a, b in _OBJN.findall(txt)}
    coher = {a: int(b) for a, b in _COHR.findall(txt)}
    supp = {a: int(b) for a, b in _SUPP.findall(txt)}
    fite = {a: int(b) for a, b in _FITE.findall(txt)}
    econ = [int(b) for _, b in _ECON.findall(txt)]
    conf = [int(b) for _, b in _CONF.findall(txt)]
    role = {a: b for a, b in _ROLE.findall(txt)}
    prims = {pid: [int(x1), int(y1), int(x2), int(y2)]
             for pid, x1, y1, x2, y2 in _PRIM.findall(txt)}
    rgns = {rid: [int(x1), int(y1), int(x2), int(y2)]
            for rid, x1, y1, x2, y2 in _RGN.findall(txt)}
    n_foreign = sum(1 for _r, rr in role.items() if rr == "foreign_object")
    n_bnd = len(set(_BND.findall(txt)))
    return dict(
        prop=prop, body_box=rgns.get("r1"), ell_box=prims.get(ellipse_id),
        body_objness=float(objn.get("r1", np.nan)),
        body_coher=float(coher.get("r1", np.nan)),
        max_support=float(max(supp.values())) if supp else np.nan,
        neg_fit_error=float(-fite.get(ellipse_id, fite.get("g1", np.nan))),
        mean_edgeconf=float(np.mean(econ)) if econ else np.nan,
        mean_conf=float(np.mean(conf)) if conf else np.nan,
        n_foreign=float(n_foreign), n_boundary=float(n_bnd),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp-dir", required=True)
    ap.add_argument("--labels", required=True, help="make_labels.py --out json (by_stem.matched50)")
    ap.add_argument("--gt-boxes", default=None, help="gt_boxes.json (crop GT) for grounding")
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ellipse-id", default="g1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--solve", action="store_true", help="add selection metrics (needs --rules/--weights)")
    ap.add_argument("--rules", default=None)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    lab = json.load(open(args.labels))
    by_stem = lab.get("by_stem", {})
    if not by_stem:
        raise SystemExit(f"{args.labels} has no by_stem (run make_labels.py with --crop-meta)")
    gtb = json.load(open(args.gt_boxes)) if args.gt_boxes else {}

    feats = {f: [] for f in FEATS}
    y, gG_body, gG_ell = [], [], []
    n_foreign_tp, n_foreign_fp, n_bnd_tp, n_bnd_fp = [], [], [], []
    n_used = n_skip = 0
    for p in sorted(glob.glob(os.path.join(args.lp_dir, "*.lp"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in by_stem:
            n_skip += 1; continue
        r = parse(p, args.ellipse_id)
        if r is None:
            n_skip += 1; continue
        tp = 1.0 if by_stem[stem]["matched50"] else 0.0
        y.append(tp)
        for f in FEATS:
            feats[f].append(r[f])
        (n_foreign_tp if tp else n_foreign_fp).append(r["n_foreign"])
        (n_bnd_tp if tp else n_bnd_fp).append(r["n_boundary"])
        if tp and stem in gtb:
            gt = gtb[stem]
            if r["body_box"]:
                gG_body.append(iou(r["body_box"], gt))
            if r["ell_box"]:
                gG_ell.append(iou(r["ell_box"], gt))
        n_used += 1
    if n_used == 0:
        raise SystemExit("no stems joined lp<->labels")
    y = np.array(y)

    disc = {}
    for f in FEATS:
        col = np.array(feats[f], float)
        a = auc(col, y)
        tpm = np.nanmean(col[y == 1]) if (y == 1).any() else np.nan
        fpm = np.nanmean(col[y == 0]) if (y == 0).any() else np.nan
        disc[f] = dict(auc=a, tp_mean=float(tpm), fp_mean=float(fpm))
    best_feat = max(disc, key=lambda k: (disc[k]["auc"] if disc[k]["auc"] == disc[k]["auc"] else 0))

    ground = dict(
        body_iou_gt=float(np.mean(gG_body)) if gG_body else None,
        ellipse_iou_gt=float(np.mean(gG_ell)) if gG_ell else None,
        n_grounded=len(gG_body))

    sel = dict(
        foreign_tp=float(np.mean(n_foreign_tp)) if n_foreign_tp else None,
        foreign_fp=float(np.mean(n_foreign_fp)) if n_foreign_fp else None,
        boundary_tp=float(np.mean(n_bnd_tp)) if n_bnd_tp else None,
        boundary_fp=float(np.mean(n_bnd_fp)) if n_bnd_fp else None)

    solve_metrics = None
    if args.solve:
        if not (args.rules and args.weights):
            raise SystemExit("--solve needs --rules and --weights")
        from learn_weights import load_rules, solve, parse_explains
        rules = load_rules(args.rules); wb = open(args.weights).read()
        bsel_tp, bsel_fp, nsel_tp, nsel_fp, fail = [], [], [], [], 0
        for p in sorted(glob.glob(os.path.join(args.lp_dir, "*.lp"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem not in by_stem:
                continue
            facts = open(p).read()
            try:
                atoms, _ = solve(rules + "\n" + facts + "\n" + wb)
                s = parse_explains(atoms)
                bsel = 1.0 if any(x in ("r1", args.ellipse_id) for x in s) else 0.0
                n = len(s)
            except Exception:
                fail += 1; continue
            tp = by_stem[stem]["matched50"]
            (bsel_tp if tp else bsel_fp).append(bsel)
            (nsel_tp if tp else nsel_fp).append(n)
        solve_metrics = dict(
            body_selected_tp=float(np.mean(bsel_tp)) if bsel_tp else None,
            body_selected_fp=float(np.mean(bsel_fp)) if bsel_fp else None,
            n_selected_tp=float(np.mean(nsel_tp)) if nsel_tp else None,
            n_selected_fp=float(np.mean(nsel_fp)) if nsel_fp else None,
            solve_fail=fail)

    report = dict(dataset=args.dataset, cls=args.cls, n=n_used,
                  n_tp=int(y.sum()), n_fp=int((1-y).sum()),
                  grounding=ground, discriminativeness=disc, best_feature=best_feat,
                  selectivity=sel, selection=solve_metrics)
    json.dump(report, open(args.out, "w"), indent=1)

    print(f"\n== {args.dataset} / {args.cls}  (n={n_used}, TP {int(y.sum())} / FP {int((1-y).sum())}) ==")
    print("  GROUNDING (matched dets):")
    print(f"    support-region IoU-GT  {ground['body_iou_gt']}")
    print(f"    ellipse IoU-GT         {ground['ellipse_iou_gt']}   (n_grounded {ground['n_grounded']})")
    print("  DISCRIMINATIVENESS (feature -> TP):")
    print(f"    {'feature':16s}  AUC    TP-mean  FP-mean")
    for f in FEATS:
        d = disc[f]
        print(f"    {f:16s}  {d['auc']:.3f}  {d['tp_mean']:8.2f} {d['fp_mean']:8.2f}")
    print(f"    best: {best_feat} (AUC {disc[best_feat]['auc']:.3f})")
    print("  SELECTIVITY (TP vs FP, from facts):")
    print(f"    foreign objs   TP {sel['foreign_tp']}  FP {sel['foreign_fp']}")
    print(f"    boundaries     TP {sel['boundary_tp']}  FP {sel['boundary_fp']}")
    if solve_metrics:
        print("  SELECTION (solved):")
        print(f"    body selected  TP {solve_metrics['body_selected_tp']}  FP {solve_metrics['body_selected_fp']}")
        print(f"    #atoms         TP {solve_metrics['n_selected_tp']}  FP {solve_metrics['n_selected_fp']}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()