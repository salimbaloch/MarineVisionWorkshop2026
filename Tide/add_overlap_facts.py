#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_overlap_facts.py -- make the ASP redundancy penalty LIVE.

object_explanation_v2.lp contains
    pair_penalty(P1,P2,redundancy,V) :- explains(P1,O), explains(P2,O),
                                        overlap(P1,P2,V), P1 != P2.
but nothing ever emits overlap/3, so weight(C,redundancy,W) is a no-op that
Optuna has been "tuning". This script appends pairwise-IoU overlap facts to
every .lp in a decomposition dir (idempotent via a marker line), so no
re-decomposition is needed.

    overlap(p1,p2,V).   % V = round(100 * IoU(box(p1), box(p2))), one order only

Boxes are taken from region_box/2 and primitive_box/2. Pairs below --min-iou
are skipped to keep grounding small.

Usage:
    python add_overlap_facts.py --lp-dir $OUT/urchin/dec [--min-iou 0.25] [--force]
"""
import os, re, glob, argparse

MARKER = "% -- overlap facts (add_overlap_facts.py) --"
_PATS = (re.compile(r"region_box\((\w+),box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\)"),
         re.compile(r"primitive_box\((\w+),box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\)"))


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def facts_for(txt, min_iou):
    boxes = {}
    for pat in _PATS:
        for m in pat.finditer(txt):
            boxes[m.group(1)] = tuple(int(m.group(i)) for i in range(2, 6))
    ids = sorted(boxes)
    out = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            v = _iou(boxes[ids[i]], boxes[ids[j]])
            if v >= min_iou:
                out.append(f"overlap({ids[i]},{ids[j]},{int(round(100 * v))}).")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp-dir", required=True)
    ap.add_argument("--min-iou", type=float, default=0.25)
    ap.add_argument("--force", action="store_true", help="re-append even if marker present")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.lp_dir, "*.lp")))
    if not paths:
        raise SystemExit(f"no .lp files in {args.lp_dir}")
    n_done = n_skip = n_facts = 0
    for p in paths:
        txt = open(p).read()
        if MARKER in txt and not args.force:
            n_skip += 1
            continue
        of = facts_for(txt, args.min_iou)
        with open(p, "a") as fh:
            fh.write("\n" + MARKER + "\n" + "\n".join(of) + ("\n" if of else ""))
        n_done += 1; n_facts += len(of)
    print(f"appended overlap facts to {n_done} files "
          f"({n_facts} facts, mean {n_facts/max(1,n_done):.1f}/file); "
          f"skipped {n_skip} already-marked (use --force to redo)")


if __name__ == "__main__":
    main()