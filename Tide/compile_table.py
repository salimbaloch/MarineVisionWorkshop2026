#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_table.py -- assemble one paper-ready results table from the per-class
reports written by faithfulness.py / oracle / localize / rescore across the
dataset x class matrix. CPU, no deps beyond stdlib.

Point it at the run root; it globs for the report jsons each stage writes and
emits (a) a wide markdown table and (b) a LaTeX booktabs table to stdout / files.

Expected per-class files under <root>/<dataset>/<class>/ (missing ones are
left blank, not fatal):
  faith_val.json                       (faithfulness.py)
  rescored_soft_<cls>_report.json      (rescore_soft.py)   -> AP baseline/rescored
  localize_<cls>.json                  (optional; localize summary if you dump it)
The AP baseline is read from rescore's report class_metrics.baseline; if absent,
pass --coco per (dataset,class) is NOT needed -- this script only compiles what
the stages already computed.

Usage:
  python compile_table.py --root $OUT --out-md $OUT/results.md --out-tex $OUT/results.tex
"""
import os, glob, json, argparse


def find(root, dataset, cls, name):
    hits = glob.glob(os.path.join(root, dataset, cls, name)) + \
           glob.glob(os.path.join(root, dataset, cls, "*" + name))
    return hits[0] if hits else None


def g(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return d if d is not None else default


def fmt(v, nd=3):
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-tex", default=None)
    args = ap.parse_args()

    # discover dataset/class dirs that have a faithfulness report
    rows = []
    for faith_path in sorted(glob.glob(os.path.join(args.root, "*", "*", "faith_val.json")) +
                             glob.glob(os.path.join(args.root, "*", "faith_val.json"))):
        rep = json.load(open(faith_path))
        d, c = rep.get("dataset", "?"), rep.get("cls", "?")
        cdir = os.path.dirname(faith_path)
        best = rep.get("best_feature")
        # AP context from oracle.json (leak-free: detector baseline + label-oracle ceilings)
        orc = None
        for cand in [os.path.join(cdir, "oracle.json")] + glob.glob(os.path.join(cdir, "*oracle*.json")):
            if os.path.exists(cand):
                try:
                    orc = json.load(open(cand)); break
                except Exception:
                    pass
        rows.append(dict(
            dataset=d, cls=c, n=rep.get("n"),
            ground_body=g(rep, "grounding", "body_iou_gt"),
            ground_ell=g(rep, "grounding", "ellipse_iou_gt"),
            disc_best=best,
            disc_auc=g(rep, "discriminativeness", best, "auc") if best else None,
            ap_base=g(orc, "baseline", "AP"),
            ap_ceil=g(orc, "ceiling_rank_iou", "AP"),
            ap75_base=g(orc, "baseline", "AP75"),
            ap75_ceil=g(orc, "ceiling_rank_iou", "AP75"),
        ))
    if not rows:
        raise SystemExit(f"no faith_val.json found under {args.root}/*/*/")

    cols = [("dataset", "dataset", 0), ("cls", "class", 0), ("n", "n", 0),
            ("ground_body", "ground(body)", 3), ("ground_ell", "ground(ell)", 3),
            ("disc_best", "best-feat", 0), ("disc_auc", "disc-AUC", 3),
            ("ap_base", "AP", 3), ("ap_ceil", "AP-ceil", 3),
            ("ap75_base", "AP75", 3), ("ap75_ceil", "AP75-ceil", 3)]

    # ---- markdown ----
    md = ["| " + " | ".join(h for _, h, _ in cols) + " |",
          "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        md.append("| " + " | ".join(fmt(r[k], nd) for k, _, nd in cols) + " |")
    md_txt = "\n".join(md)
    print(md_txt)

    # ---- latex ----
    tex = ["\\begin{tabular}{ll" + "r" * (len(cols) - 2) + "}", "\\toprule",
           " & ".join(h for _, h, _ in cols) + " \\\\", "\\midrule"]
    last_ds = None
    for r in rows:
        if last_ds is not None and r["dataset"] != last_ds:
            tex.append("\\midrule")
        last_ds = r["dataset"]
        tex.append(" & ".join(fmt(r[k], nd).replace("_", "\\_") for k, _, nd in cols) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    tex_txt = "\n".join(tex)

    if args.out_md:
        open(args.out_md, "w").write(md_txt + "\n")
        print(f"\nwrote {args.out_md}")
    if args.out_tex:
        open(args.out_tex, "w").write(tex_txt + "\n")
        print(f"wrote {args.out_tex}")


if __name__ == "__main__":
    main()