
import os, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from box_regressor import (RefineSet, JitterView, BoxRegressor, decode, blend,
                           giou_loss, split_indices, evaluate, report, CHANNELS)


# --------------------------------------------------------- no-atoms variant
class NoAtomBoxRegressor(nn.Module):
    """Identical trunk/head to BoxRegressor but the primitive branch is fed ZEROS,
    so only the DINOv3 embedding + proposal can inform the offset. Same param
    count and capacity as the full model -> a fair ablation."""
    def __init__(self, emb_dim, d=128):
        super().__init__()
        self.inner = BoxRegressor(emb_dim, d=d)

    def forward(self, emb, prim):
        return self.inner(emb, torch.zeros_like(prim))


def train_loop(net, tr_dl, va_dl, dev, epochs, lr, wd, tag, log_every=5):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        net.train(); tot = nb = 0
        for emb, prim, prop, gt in tr_dl:
            emb, prim, prop, gt = emb.to(dev), prim.to(dev), prop.to(dev), gt.to(dev)
            loss, _ = giou_loss(decode(prop, net(emb, prim)), gt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        if ep == 0 or (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"    [{tag}] epoch {ep+1:>3}/{epochs}  train GIoU loss {tot/max(1,nb):.4f}")
    return net


def mode_overfit(ds, args, dev):
    """Train to convergence on a handful of FIXED samples (no jitter, high epochs)."""
    idx = list(range(min(args.batch_size, len(ds))))
    dl = DataLoader(Subset(ds, idx), batch_size=len(idx), shuffle=False, num_workers=0)
    print(f"OVERFIT-BATCH: {len(idx)} fixed samples, {args.overfit_epochs} epochs, no jitter")
    net = BoxRegressor(ds[0][0].shape[0]).to(dev)
    train_loop(net, dl, dl, dev, args.overfit_epochs, args.lr, 0.0, "overfit", log_every=50)
    m = evaluate(net, dl, dev, alpha=1.0)
    print(f"\n  final on the SAME batch:  det IoU {m['det']:.3f} -> ref IoU {m['ref']:.3f} "
          f"(delta {m['ref']-m['det']:+.3f})")
    print("  VERDICT: " + (
        "loss ~0 and IoU->~1  => net CAN fit; fault is representation/data, not the model."
        if m['ref'] > 0.95 else
        "cannot reach IoU~1 on a handful of samples => ARCHITECTURE/TRAINING bug (Layer 1-arch). "
        "Investigate offset param / loss / lr before touching atoms."))


def mode_ablate(ds, args, dev):
    """Full model vs no-atoms, identical split & recipe."""
    tr_idx, va_idx = split_indices(ds.items, args.val_frac, args.split_by, args.seed)
    np.random.seed(args.seed)
    tr_view = (JitterView(ds, tr_idx, args.jitter_shift, args.jitter_scale, args.jitter_repeat)
               if args.jitter_shift > 0 else Subset(ds, tr_idx))
    tr_dl = DataLoader(tr_view, batch_size=args.batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(Subset(ds, va_idx), batch_size=args.batch, shuffle=False, num_workers=0)
    emb_dim = ds[0][0].shape[0]

    print(f"ABLATE: train {len(tr_idx)} / val {len(va_idx)}   epochs={args.epochs}\n")
    results = {}
    for tag, Net in (("FULL (emb+atoms)", BoxRegressor), ("NO-ATOMS (emb only)", NoAtomBoxRegressor)):
        torch.manual_seed(args.seed)
        net = Net(emb_dim).to(dev)
        train_loop(net, tr_dl, va_dl, dev, args.epochs, args.lr, args.weight_decay, tag)
        # alpha=1 to see the RAW regressor contribution (no gate masking it)
        m = evaluate(net, va_dl, dev, alpha=1.0)
        results[tag] = m
        report(tag, m)
        print()

    f, n = results["FULL (emb+atoms)"], results["NO-ATOMS (emb only)"]
    gap = f["ref"] - n["ref"]
    print(f"ATOM CONTRIBUTION (val IoU, alpha=1): full {f['ref']:.3f} vs no-atoms {n['ref']:.3f} "
          f"= {gap:+.3f}")
    print("  VERDICT: " + (
        f"atoms add {gap:+.3f} IoU -> the channels DO carry signal; refine the representation to amplify."
        if gap > 0.01 else
        "atoms add ~nothing -> the rendered channels are currently uninformative. "
        "This is your Sec 4.3 result AND the case for stroke rendering / extent coords."))


def mode_dump(ds, args, dev):
    """Write per-crop panels: each selected-atom channel side by side, so you can
    SEE whether boundary/edge atoms localize the edge or are filled blobs."""
    import cv2
    os.makedirs(args.out_dir, exist_ok=True)
    n = min(args.num, len(ds)) if args.num else len(ds)
    for i in range(n):
        emb, prim, prop, gt = ds[i]
        stem = ds.items[i][0]
        prim = prim.numpy()                       # (K,S,S) in [0,1]
        tiles = []
        for c in range(prim.shape[0]):
            t = (prim[c] * 255).astype(np.uint8)
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            cv2.putText(t, CHANNELS[c], (3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 240, 0), 1, cv2.LINE_AA)
            cv2.rectangle(t, (0, 0), (t.shape[1]-1, t.shape[0]-1), (60, 60, 60), 1)
            tiles.append(t)
        strip = cv2.hconcat(tiles)
        cov = (prim.max(0) > 0).mean()            # fraction of crop any atom covers
        hdr = np.full((22, strip.shape[1], 3), 24, np.uint8)
        cv2.putText(hdr, f"{stem}   atom-coverage={cov:.2f}   [filled=blocky, thin=informative]",
                    (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_chan.png"), cv2.vconcat([hdr, strip]))
    print(f"wrote {n} channel panels -> {args.out_dir}")
    print("  look at boundary/edge_fragment columns: solid filled rectangles = the "
          "rendering defect; thin arcs following the object edge = informative.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["overfit-batch", "ablate", "dump-channels"])
    ap.add_argument("--rules", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--lp-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--S", type=int, default=64)
    # overfit
    ap.add_argument("--batch-size", type=int, default=8, help="samples in the overfit batch")
    ap.add_argument("--overfit-epochs", type=int, default=400)
    # ablate / shared training
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--split-by", choices=["detection", "image"], default="image")
    ap.add_argument("--jitter-shift", type=float, default=0.15)
    ap.add_argument("--jitter-scale", type=float, default=0.25)
    ap.add_argument("--jitter-repeat", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    # dump
    ap.add_argument("--out-dir", default="channels")
    ap.add_argument("--num", type=int, default=12)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = RefineSet(args.lp_dir, args.emb_dir, args.gt, args.rules, args.weights, args.cls, args.S)
    print(f"[{args.cls}] {len(ds)} detections   device={dev}\n")
    {"overfit-batch": mode_overfit, "ablate": mode_ablate, "dump-channels": mode_dump}[args.mode](
        ds, args, dev)


if __name__ == "__main__":
    main()