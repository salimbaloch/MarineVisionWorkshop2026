#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_candidates.py -- candidate-box rows for the primitive selector.

For every decomposed detection, enumerate candidate boxes (the proposal itself
PLUS each primitive box), attach per-candidate features, and label each with its
IoU-to-GT. The selector (train_selector.py) learns to predict that IoU from
features and, at inference, swaps in the best-predicted candidate -- but only
through a margin gate, since the localize probe showed 82/253 loose dets have NO
candidate better than the proposal and every wrong swap costs AP.

CRITICAL design point (from the oracle): the PROPOSAL is itself a candidate.
The task is not "pick the best primitive" but "pick the best box INCLUDING doing
nothing". Predicting the proposal's own quality is how the gate learns when to
leave a box alone.

Candidate sources (crop frame, matching localize_ceiling.py):
  proposal          detector box (the do-nothing option)
  ellipse           primitive_box(g1)   [id configurable]
  region_body       region_box(r1)
  boundary_i        each boundary_primitive's primitive_box
  support@t         largest-CC bbox of (support>=t), if --support-dir given
  union_selected    union of ASP-selected boxes, if --solve given

Per-candidate features (all cheap, from facts + geometry; no GT):
  cand_type onehot  proposal/ellipse/region/boundary/support/union
  iou_to_proposal   overlap with the detector box
  area_ratio        cand area / proposal area
  dcx,dcy           center offset from proposal (normalized by proposal size)
  log_area, aspect  candidate geometry
  support,conf      atom semantic_support / confidence (0 for proposal)
  edge_conf         atom edge_confidence (0 where n/a)
  fit_error         ellipse/shape fit error (0 where n/a)
  body_objness/coher, max_support   detection-level context (same on all rows)
  is_selected       1 if this box was ASP-selected (needs --solve)
  n_cand            number of candidates for this detection

Label:
  iou_gt            IoU(candidate box, matched GT)          [regression target]
  (quality_coco     mean over IoU>=[.5:.05:.95] if --coco-quality; harder target)

Output: an .npz with X (rows x feats), y (iou_gt), and parallel arrays
  stem, cand_name, cand_box_img (xywh, for the swap at inference), plus feat_names
  and the per-row proposal box. Only detections with a GT match are emitted
  (selector is trained/evaluated on matched dets, like best-prim-swap).

Usage:
  python build_candidates.py --lp-dir $DEC_VAL \
    --crop-meta $DEC_VAL/crop_meta.json --gt-boxes $DEC_VAL/gt_boxes.json \
    --class animal_urchin --support-dir $DEC_VAL/support \
    --out $OUT/urchin/cand_val.npz
    [--solve --rules $ASP --weights $WEIGHTS]

Deps: numpy; cv2 (only for --support-dir); clingo/learn_weights (only for --solve).
"""
import os, re, glob, json, argparse
import numpy as np

_PROP = re.compile(r"proposal_box\(\w+,box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\)")
_PRIM = re.compile(r"primitive_box\((\w+),box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\)")
_RGN = re.compile(r"region_box\((\w+),box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\)")
_ROLE = re.compile(r"region_role\((\w+),(\w+)\)")
_BND = re.compile(r"boundary_primitive\((\w+)\)")
_SUPP = re.compile(r"semantic_support\((\w+),(-?\d+)\)")
_CONF = re.compile(r"confidence\((\w+),(-?\d+)\)")
_ECON = re.compile(r"edge_confidence\((\w+),(-?\d+)\)")
_FITE = re.compile(r"fit_error\((\w+),(-?\d+)\)")
_OBJN = re.compile(r"region_objectness\((\w+),(-?\d+)\)")
_COHR = re.compile(r"region_coherence\((\w+),(-?\d+)\)")

CAND_TYPES = ["proposal", "ellipse", "region", "boundary", "support", "union"]
FEAT_NAMES = ([f"type_{t}" for t in CAND_TYPES] +
              ["iou_to_proposal", "area_ratio", "dcx", "dcy", "log_area", "aspect",
               "support", "conf", "edge_conf", "fit_error",
               "body_objness", "body_coher", "max_support", "is_selected", "n_cand"])


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def support_boxes(sup_path, thresholds):
    try:
        import cv2
        sup = np.load(sup_path)["sup"].astype(np.float32) / 255.0
    except Exception:
        return {}
    out = {}
    for t in thresholds:
        m = (sup >= t).astype(np.uint8)
        if m.sum() < 4:
            continue
        n, _l, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            continue
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, _ = stats[big]
        out[f"support@{t:.2f}"] = [int(x), int(y), int(x+w), int(y+h)]
    return out


def parse_candidates(path, ellipse_id, sup_boxes, selected):
    """Return (proposal, [(name, box, type, feat_atoms), ...]) in crop frame."""
    txt = open(path).read()
    mp = _PROP.search(txt)
    if not mp:
        return None
    prop = [int(v) for v in mp.groups()]
    role = {a: b for a, b in _ROLE.findall(txt)}
    supp = {a: int(b) for a, b in _SUPP.findall(txt)}
    conf = {a: int(b) for a, b in _CONF.findall(txt)}
    econ = {a: int(b) for a, b in _ECON.findall(txt)}
    fite = {a: int(b) for a, b in _FITE.findall(txt)}
    objn = {a: int(b) for a, b in _OBJN.findall(txt)}
    cohr = {a: int(b) for a, b in _COHR.findall(txt)}
    prims = {pid: [int(x1), int(y1), int(x2), int(y2)]
             for pid, x1, y1, x2, y2 in _PRIM.findall(txt)}
    rgns = {rid: [int(x1), int(y1), int(x2), int(y2)]
            for rid, x1, y1, x2, y2 in _RGN.findall(txt)}
    bnds = set(_BND.findall(txt))

    body_obj = float(objn.get("r1", 0))
    body_coh = float(cohr.get("r1", 0))
    max_sup = float(max(supp.values())) if supp else 0.0

    cands = []
    # proposal: the do-nothing candidate
    cands.append(("proposal", prop, "proposal",
                  dict(support=0, conf=0, edge_conf=0, fit_error=0)))
    # ellipse
    if ellipse_id in prims:
        cands.append(("ellipse", prims[ellipse_id], "ellipse",
                      dict(support=supp.get(ellipse_id, 0), conf=conf.get(ellipse_id, 0),
                           edge_conf=econ.get(ellipse_id, 0), fit_error=fite.get(ellipse_id, 0))))
    # body region
    if "r1" in rgns:
        cands.append(("region_body", rgns["r1"], "region",
                      dict(support=supp.get("r1", 0), conf=conf.get("r1", 0),
                           edge_conf=0, fit_error=0)))
    # boundary primitives
    for bid in sorted(bnds):
        if bid in prims:
            cands.append((f"boundary_{bid}", prims[bid], "boundary",
                          dict(support=supp.get(bid, 0), conf=conf.get(bid, 0),
                               edge_conf=econ.get(bid, 0), fit_error=0)))
    # support-threshold boxes
    for name, box in (sup_boxes or {}).items():
        cands.append((name, box, "support",
                      dict(support=int(max_sup), conf=0, edge_conf=0, fit_error=0)))
    # union of ASP-selected boxes
    if selected:
        sb = [prims[p] for p in selected if p in prims] + \
             [rgns[p] for p in selected if p in rgns]
        if sb:
            ub = [min(b[0] for b in sb), min(b[1] for b in sb),
                  max(b[2] for b in sb), max(b[3] for b in sb)]
            cands.append(("union_selected", ub, "union",
                          dict(support=0, conf=0, edge_conf=0, fit_error=0)))

    ctx = dict(body_objness=body_obj, body_coher=body_coh, max_support=max_sup,
               selected=selected or set())
    return prop, cands, ctx


def featurize(prop, box, ctype, atoms, ctx, n_cand, name):
    pw = max(1.0, prop[2]-prop[0]); ph = max(1.0, prop[3]-prop[1])
    cw = max(1.0, box[2]-box[0]); ch = max(1.0, box[3]-box[1])
    pcx, pcy = (prop[0]+prop[2])/2, (prop[1]+prop[3])/2
    ccx, ccy = (box[0]+box[2])/2, (box[1]+box[3])/2
    onehot = [1.0 if ctype == t else 0.0 for t in CAND_TYPES]
    is_sel = 1.0 if name in ctx["selected"] else 0.0
    feats = onehot + [
        iou(box, prop),
        (cw*ch) / (pw*ph),
        (ccx-pcx)/pw, (ccy-pcy)/ph,
        float(np.log1p(cw*ch)),
        float(np.clip(cw/ch, 0.1, 10.0)),
        float(atoms["support"]), float(atoms["conf"]),
        float(atoms["edge_conf"]), float(atoms["fit_error"]),
        ctx["body_objness"], ctx["body_coher"], ctx["max_support"],
        is_sel, float(n_cand),
    ]
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp-dir", required=True)
    ap.add_argument("--crop-meta", required=True)
    ap.add_argument("--gt-boxes", required=True)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--ellipse-id", default="g1")
    ap.add_argument("--support-dir", default=None)
    ap.add_argument("--sup-thresholds", default="0.3,0.5,0.7")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--rules", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cmeta = json.load(open(args.crop_meta))
    gtb = json.load(open(args.gt_boxes))
    sup_thrs = [float(t) for t in args.sup_thresholds.split(",")]

    solve_fn = None
    if args.solve:
        if not (args.rules and args.weights):
            raise SystemExit("--solve needs --rules and --weights")
        from learn_weights import load_rules, solve, parse_explains
        rules = load_rules(args.rules); wb = open(args.weights).read()
        def solve_fn(facts):
            try:
                atoms, _ = solve(rules + "\n" + facts + "\n" + wb)
                return parse_explains(atoms)
            except Exception:
                return set()

    X, y, stems, names, boxes_img, prop_ious = [], [], [], [], [], []
    n_det = n_skip = 0
    for p in sorted(glob.glob(os.path.join(args.lp_dir, "*.lp"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in gtb or stem not in cmeta:
            n_skip += 1; continue
        sb = support_boxes(os.path.join(args.support_dir, stem + ".npz"), sup_thrs) \
            if args.support_dir else {}
        sel = solve_fn(open(p).read()) if solve_fn else set()
        parsed = parse_candidates(p, args.ellipse_id, sb, sel)
        if parsed is None:
            n_skip += 1; continue
        prop, cands, ctx = parsed
        gt = gtb[stem]
        cm = cmeta[stem]; ox, oy = cm[1], cm[2]
        n_c = len(cands)
        pj = iou(prop, gt)
        for name, box, ctype, atoms in cands:
            X.append(featurize(prop, box, ctype, atoms, ctx, n_c, name))
            y.append(iou(box, gt))
            stems.append(stem); names.append(name)
            # image-frame xywh for the eventual swap
            boxes_img.append([box[0]+ox, box[1]+oy, (box[2]-box[0]), (box[3]-box[1])])
            prop_ious.append(pj)
        n_det += 1

    if n_det == 0:
        raise SystemExit("no matched detections with candidates")
    X = np.asarray(X, np.float32); y = np.asarray(y, np.float32)
    np.savez_compressed(args.out, X=X, y=y,
                        stem=np.array(stems), cand=np.array(names),
                        box_img=np.asarray(boxes_img, np.float32),
                        prop_iou=np.asarray(prop_ious, np.float32),
                        feat_names=np.array(FEAT_NAMES))
    # quick sanity: per-candidate-type mean IoU (should echo localize's per-source table)
    print(f"[candidates] {n_det} dets -> {len(y)} rows ({n_skip} skipped)  -> {args.out}")
    print(f"  {'cand_type':12s}  n     mean IoU_gt")
    cand_arr = np.array(names)
    typ = np.array([n.split("@")[0].split("_")[0] for n in names])
    for t in CAND_TYPES:
        sel_t = typ == t if t != "region" else np.array(["region" in n for n in names])
        if sel_t.sum():
            print(f"  {t:12s}  {int(sel_t.sum()):5d}  {y[sel_t].mean():.3f}")
    # how often SOME non-proposal candidate beats the proposal (oracle upper bound)
    by_stem = {}
    for i, s in enumerate(stems):
        by_stem.setdefault(s, []).append(i)
    beat = sum(1 for s, idx in by_stem.items()
               if max(y[i] for i in idx) > y[[i for i in idx if names[i] == "proposal"][0]] + 1e-6)
    print(f"  dets where some candidate beats proposal: {beat}/{len(by_stem)} "
          f"({100.0*beat/len(by_stem):.1f}%)  [selector oracle ceiling]")


if __name__ == "__main__":
    main()