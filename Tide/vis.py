
import os, glob, json, argparse, random
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import core
from learn_weights import load_rules, solve, parse_facts, parse_explains
from box_regressor import render_channels, BoxRegressor, decode


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def build_stem2info(pred, gt_json, target, score_thr, box_format):
    """stem -> (image_file, image-coord box). Mirrors m4decompose's detection loop so
    the stem numbering matches (k = index over filtered dets; skips don't renumber)."""
    by_img = core.load_predictions(pred, box_format)
    id2file = core.load_gt_index(gt_json)[0]
    info = {}
    for image_id, dets in by_img.items():
        dets = [d for d in dets if d["cat"] == target and d["score"] >= score_thr]
        if not dets:
            continue
        fname = id2file.get(image_id)
        if not fname:
            continue
        base = os.path.splitext(os.path.basename(fname))[0]
        for k, d in enumerate(dets):
            info[f"{base}_det{k}"] = (fname, d["box"], d["score"])
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--weights", required=True, help="weights_<class>.lp")
    ap.add_argument("--pt", required=True, help="box_regressor_<class>.pt")
    ap.add_argument("--lp-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--gt-boxes", required=True, help="gt_boxes.json (crop coords)")
    ap.add_argument("--class", dest="cls", required=True)
    # re-crop inputs (same as the decompose run)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt-json", required=True, help="instances_val.json")
    ap.add_argument("--images", required=True)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--box-format", choices=["xywh", "xyxy"], default="xywh")
    # display
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--exclude-substr", default=None, help="skip stems containing this (held-out seq)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rules = load_rules(args.rules)
    wb = open(args.weights).read()
    gt = {k: tuple(v) for k, v in json.load(open(args.gt_boxes)).items()}
    name2cat = core.load_gt_index(args.gt_json)[1]
    target = name2cat[args.cls]
    info = build_stem2info(args.pred, args.gt_json, target, args.score_thr, args.box_format)

    stems = []
    for p in sorted(glob.glob(os.path.join(args.lp_dir, "*.lp"))):
        s = os.path.splitext(os.path.basename(p))[0]
        if s in gt and s in info and os.path.exists(os.path.join(args.emb_dir, s + ".npz")):
            if args.exclude_substr and args.exclude_substr in s:
                continue
            stems.append(s)
    if not stems:
        raise SystemExit("no stems with .lp + emb + gt + prediction info")
    random.Random(args.seed).shuffle(stems)
    stems = stems[:args.n]

    emb_dim = np.load(os.path.join(args.emb_dir, stems[0] + ".npz"))["emb"].shape[0]
    net = BoxRegressor(emb_dim)
    net.load_state_dict(torch.load(args.pt, map_location="cpu"))
    net.eval()

    cols = args.cols
    rows = (len(stems) + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    ax = np.atleast_1d(ax).ravel()
    det_ious, ref_ious = [], []

    for i, s in enumerate(stems):
        facts = open(os.path.join(args.lp_dir, s + ".lp")).read()
        meta = parse_facts(facts)
        atoms, _ = solve(rules + "\n" + facts + "\n" + wb)
        sel = parse_explains(atoms)
        prim = render_channels(facts, sel, meta, args.S)
        emb = np.load(os.path.join(args.emb_dir, s + ".npz"))["emb"].astype(np.float32)
        emb = cv2.resize(emb.transpose(1, 2, 0), (args.S, args.S),
                         interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        with torch.no_grad():
            off = net(torch.from_numpy(emb)[None], torch.from_numpy(prim)[None])
            ref = decode(torch.tensor(meta["proposal"], dtype=torch.float32)[None], off)[0].numpy()

        det = np.asarray(meta["proposal"], float)
        gtb = np.asarray(gt[s], float)
        fname, box, score = info[s]
        img = np.asarray(Image.open(os.path.join(args.images, fname)).convert("RGB"))
        crop, (ox, oy, _, _) = core.crop_with_margin(img, box, args.margin)

        a = ax[i]
        a.imshow(crop)
        ov = cv2.resize(prim.max(0), (crop.shape[1], crop.shape[0]))          # selected-atom footprint
        a.imshow(ov, cmap="viridis", alpha=0.22)
        for b, c in [(det, "yellow"), (ref, "lime"), (gtb, "#00ff88")]:
            a.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec=c, lw=2))
        di, ri = iou(det, gtb), iou(ref, gtb)
        det_ious.append(di); ref_ious.append(ri)
        a.set_title(f"{s}\ndet {di:.2f} -> ref {ri:.2f}", fontsize=8,
                    color=("green" if ri > di + 1e-3 else "red" if ri < di - 1e-3 else "black"))
        a.axis("off")

    for j in range(len(stems), len(ax)):
        ax[j].axis("off")

    md, mr = float(np.mean(det_ious)), float(np.mean(ref_ious))
    win = sum(r > d + 1e-3 for d, r in zip(det_ious, ref_ious))
    fig.suptitle(f"{args.cls}  |  yellow=det  lime=ref  green=gt  |  "
                 f"mean IoU  det {md:.3f} -> ref {mr:.3f}  |  improved {win}/{len(stems)}"
                 + ("  [TRAINING CROPS]" if not args.exclude_substr else f"  [held out: {args.exclude_substr}]"),
                 fontsize=11)
    fig.tight_layout()
    out = args.out or f"vis_{args.cls}.png"
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"{args.cls}: mean IoU  det {md:.3f} -> ref {mr:.3f}   improved {win}/{len(stems)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()