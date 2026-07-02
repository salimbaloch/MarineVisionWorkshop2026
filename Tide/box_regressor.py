
import os, re, glob, json, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from learn_weights import load_rules, solve, parse_facts, parse_explains


# ------------------------------------------------------------------ fact geometry
_ELL = re.compile(r"ellipse\((\w+),point\((-?\d+),(-?\d+)\),axes\((-?\d+),(-?\d+)\),angle\((-?\d+)\)\)")
_RECT = re.compile(r"rect\((\w+),point\((-?\d+),(-?\d+)\),size\((-?\d+),(-?\d+)\),angle\((-?\d+)\)\)")
_CONF = re.compile(r"confidence\((\w+),(-?\d+)\)")
_CROP = re.compile(r"crop_size\([^,]+,size\((\d+),(\d+)\)\)")
_TYPE = re.compile(r"primitive_type\((\w+),(\w+)\)")
_BSRC = re.compile(r"boundary_source\((\w+),(\w+)\)")


def _crop_hw(facts, meta):
    m = _CROP.search(facts)
    if m:
        return int(m.group(1)), int(m.group(2))          # (H, W)
    xs = [b[2] for b in meta["boxes"].values()] + ([meta["proposal"][2]] if meta["proposal"] else [])
    ys = [b[3] for b in meta["boxes"].values()] + ([meta["proposal"][3]] if meta["proposal"] else [])
    return (max(ys) + 1, max(xs) + 1) if xs else (64, 64)


# channel layout: which atom kinds map to which confidence channel
CHANNELS = ["body_region", "ellipse", "rect", "boundary", "edge_fragment"]

def render_channels(facts, selected, meta, S=64):
    """Render each SELECTED primitive into an (len(CHANNELS), S, S) map; intensity =
    the atom's confidence/100, placed in the crop frame rescaled to S x S."""
    H, W = _crop_hw(facts, meta)
    sx, sy = S / max(W, 1), S / max(H, 1)
    conf = {m.group(1): int(m.group(2)) / 100.0 for m in _CONF.finditer(facts)}
    ptype = {m.group(1): m.group(2) for m in _TYPE.finditer(facts)}
    bsrc = {m.group(1): m.group(2) for m in _BSRC.finditer(facts)}
    ell = {m.group(1): tuple(int(m.group(i)) for i in range(2, 7)) for m in _ELL.finditer(facts)}
    rect = {m.group(1): tuple(int(m.group(i)) for i in range(2, 7)) for m in _RECT.finditer(facts)}
    ch = {c: np.zeros((S, S), np.float32) for c in CHANNELS}

    def fill(mask_u8, name, cid):
        m = cv2.resize(mask_u8, (S, S), interpolation=cv2.INTER_NEAREST) > 0
        ch[name][m] = np.maximum(ch[name][m], conf.get(cid, 0.5))

    for pid in selected:
        canvas = np.zeros((H, W), np.uint8)
        if pid in ell:                                   # ellipse primitive
            cx, cy, d1, d2, ang = ell[pid]
            cv2.ellipse(canvas, (cx, cy), (max(1, d1 // 2), max(1, d2 // 2)),
                        float(ang), 0, 360, 255, -1)
            fill(canvas, "ellipse", pid); continue
        if pid in rect:                                  # rect primitive
            cx, cy, w, h, ang = rect[pid]
            box = cv2.boxPoints(((cx, cy), (w, h), float(ang))).astype(np.int32)
            cv2.fillPoly(canvas, [box], 255)
            fill(canvas, "rect", pid); continue
        if pid not in meta["boxes"]:
            continue
        x1, y1, x2, y2 = meta["boxes"][pid]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, -1)
        if pid == meta["body"]:
            fill(canvas, "body_region", pid)
        elif pid in meta["boundary_ids"]:
            fill(canvas, "edge_fragment" if bsrc.get(pid) == "edge" else "boundary", pid)
    # scale coords already handled by resize; return C,H,W
    return np.stack([ch[c] for c in CHANNELS], 0)


# ------------------------------------------------------------------ dataset
class RefineSet(Dataset):
    def __init__(self, lp_dir, emb_dir, gt_json, rules, weights_lp, cls, S=64):
        self.S = S
        self.rules = load_rules(rules)
        self.wb = open(weights_lp).read()
        self.emb_dir = emb_dir
        gt = {k: tuple(v) for k, v in json.load(open(gt_json)).items()}
        self.items = []
        for p in sorted(glob.glob(os.path.join(lp_dir, "*.lp"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            emb_p = os.path.join(emb_dir, stem + ".npz")
            if stem not in gt or not os.path.exists(emb_p):
                continue
            facts = open(p).read()
            meta = parse_facts(facts)
            if meta["cls"] != cls or meta["proposal"] is None:
                continue
            self.items.append((stem, facts, meta, gt[stem], emb_p))
        if not self.items:
            raise SystemExit("no samples: check lp_dir / emb_dir / gt / class")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, facts, meta, gtb, emb_p = self.items[i]
        atoms, _ = solve(self.rules + "\n" + facts + "\n" + self.wb)
        selected = parse_explains(atoms)
        prim = render_channels(facts, selected, meta, self.S)                 # (K,S,S)
        emb = np.load(emb_p)["emb"].astype(np.float32)                        # (Ce,gh,gw)
        emb = cv2.resize(emb.transpose(1, 2, 0), (self.S, self.S),
                         interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)   # (Ce,S,S)
        return (torch.from_numpy(emb), torch.from_numpy(prim),
                torch.tensor(meta["proposal"], dtype=torch.float32),
                torch.tensor(gtb, dtype=torch.float32))


# ------------------------------------------------------------------ model
class BoxRegressor(nn.Module):
    def __init__(self, emb_dim, n_prim=len(CHANNELS), d=128):
        super().__init__()
        self.emb_proj = nn.Conv2d(emb_dim, d, 1)
        self.prim_proj = nn.Conv2d(n_prim, d, 3, padding=1)
        self.trunk = nn.Sequential(
            nn.Conv2d(2 * d, d, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, 4))

    def forward(self, emb, prim):
        x = torch.cat([self.emb_proj(emb), self.prim_proj(prim)], 1)
        x = self.trunk(x).flatten(1)
        return self.head(x)                                    # (B,4) = (dcx,dcy,dw,dh)


def decode(proposal, off):
    px1, py1, px2, py2 = proposal.unbind(-1)
    pw = (px2 - px1).clamp(min=1); ph = (py2 - py1).clamp(min=1)
    pcx = (px1 + px2) / 2; pcy = (py1 + py2) / 2
    dcx, dcy, dw, dh = off.unbind(-1)
    cx = pcx + dcx * pw; cy = pcy + dcy * ph
    w = pw * torch.exp(dw.clamp(-2, 2)); h = ph * torch.exp(dh.clamp(-2, 2))
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def giou_loss(pred, gt, eps=1e-7):
    x1 = torch.max(pred[:, 0], gt[:, 0]); y1 = torch.max(pred[:, 1], gt[:, 1])
    x2 = torch.min(pred[:, 2], gt[:, 2]); y2 = torch.min(pred[:, 3], gt[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    ap = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    ag = (gt[:, 2] - gt[:, 0]).clamp(min=0) * (gt[:, 3] - gt[:, 1]).clamp(min=0)
    union = ap + ag - inter + eps
    iou = inter / union
    cx1 = torch.min(pred[:, 0], gt[:, 0]); cy1 = torch.min(pred[:, 1], gt[:, 1])
    cx2 = torch.max(pred[:, 2], gt[:, 2]); cy2 = torch.max(pred[:, 3], gt[:, 3])
    carea = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + eps
    giou = iou - (carea - union) / carea
    return (1 - giou).mean(), iou.mean().item()


# ------------------------------------------------------------------ split + eval (§4.2)
def split_indices(items, val_frac, split_by, seed):

    import random as _r
    n = len(items)
    n_test = max(1, int(round(val_frac * n)))
    if split_by == "image":
        from collections import defaultdict
        groups = defaultdict(list)
        for i, it in enumerate(items):
            groups[it[0].rsplit("_det", 1)[0]].append(i)
        keys = list(groups); _r.Random(seed).shuffle(keys)
        test, acc = [], 0
        for k in keys:
            if acc >= n_test:
                break
            test += groups[k]; acc += len(groups[k])
        tset = set(test)
        return [i for i in range(n) if i not in tset], sorted(tset)
    idx = list(range(n)); _r.Random(seed).shuffle(idx)
    return idx[n_test:], idx[:n_test]


def _iou_np(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@torch.no_grad()
def evaluate(net, loader, dev):
    """Per-detection: baseline IoU(det,gt) vs refined IoU(ref,gt), area ratio, %improved."""
    net.eval()
    ds = rs = ar = 0.0; imp = n = 0
    for emb, prim, prop, gt in loader:
        emb, prim = emb.to(dev), prim.to(dev)
        ref = decode(prop.to(dev), net(emb, prim)).cpu().numpy()
        prop_np, gt_np = prop.numpy(), gt.numpy()
        for k in range(len(gt_np)):
            di = _iou_np(prop_np[k], gt_np[k]); ri = _iou_np(ref[k], gt_np[k])
            ds += di; rs += ri; imp += int(ri > di + 1e-6); n += 1
            ad = max(0.0, prop_np[k][2]-prop_np[k][0]) * max(0.0, prop_np[k][3]-prop_np[k][1])
            arr = max(0.0, ref[k][2]-ref[k][0]) * max(0.0, ref[k][3]-ref[k][1])
            ar += (arr / ad) if ad > 0 else 1.0
    return dict(n=n, det=ds/n, ref=rs/n, improved=imp, area_ratio=ar/n)


def report(tag, m):
    print(f"  [{tag:>5}] n={m['n']:<5}  IoU  det {m['det']:.3f} -> ref {m['ref']:.3f}  "
          f"(delta {m['ref']-m['det']:+.3f})   improved {m['improved']}/{m['n']} "
          f"({100*m['improved']/m['n']:.0f}%)   mean area ref/det {m['area_ratio']:.2f} "
          f"({'shrink' if m['area_ratio']<1 else 'grow'} {abs(1-m['area_ratio'])*100:.0f}%)")


# ------------------------------------------------------------------ train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--weights", required=True, help="weights_<class>.lp from learn_weights.py")
    ap.add_argument("--lp-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.2, help="held-out test fraction of val")
    ap.add_argument("--split-by", choices=["detection", "image"], default="detection",
                    help="detection = plain random (matches the leaky baseline regime); "
                         "image = whole frames on one side (no same-frame leakage)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from torch.utils.data import Subset
    ds = RefineSet(args.lp_dir, args.emb_dir, args.gt, args.rules, args.weights, args.cls, args.S)
    tr_idx, te_idx = split_indices(ds.items, args.val_frac, args.split_by, args.seed)
    tr_dl = DataLoader(Subset(ds, tr_idx), batch_size=args.batch, shuffle=True, num_workers=args.workers)
    te_dl = DataLoader(Subset(ds, te_idx), batch_size=args.batch, shuffle=False, num_workers=args.workers)

    emb_dim = ds[0][0].shape[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = BoxRegressor(emb_dim).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[{args.cls}] {len(ds)} samples -> train {len(tr_idx)} / test {len(te_idx)} "
          f"(split_by={args.split_by}, seed={args.seed})  emb_dim={emb_dim}  device={dev}")

    for ep in range(args.epochs):
        net.train(); tot = 0.0; iou_sum = 0.0; n = 0
        for emb, prim, prop, gt in tr_dl:
            emb, prim, prop, gt = emb.to(dev), prim.to(dev), prop.to(dev), gt.to(dev)
            ref = decode(prop, net(emb, prim))
            loss, iou = giou_loss(ref, gt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * emb.size(0); iou_sum += iou * emb.size(0); n += emb.size(0)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1:3d}  train loss {tot/n:.4f}  train IoU {iou_sum/n:.4f}")

    
    print(f"\n=== {args.cls}: localization vs DEIMv2 proposals ===")
    report("train", evaluate(net, tr_dl, dev))
    report("TEST",  evaluate(net, te_dl, dev))
    print("  (TEST is held out; the train row is only to gauge the generalization gap.)")

    out = args.out or f"box_regressor_{args.cls}.pt"
    torch.save(net.state_dict(), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()