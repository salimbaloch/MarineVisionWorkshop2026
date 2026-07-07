#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_predictions.py  --  run a trained DEIMv2 detector over a COCO split and
write predictions in COCO-results format:  [{image_id, category_id, bbox(xywh), score}]

Generalized from the RUOD-R hardcoded version so it can be pointed at ANY split
(train OR val) via CLI. For the SeaClear refinement result you want the *val*
split (detector-unseen -> realistic proposals); train is available too but its
proposals are overfit to the detector, so don't train the regressor on them.

Example (SeaClear val):
    python dump_predictions.py \
        --deim-root /srv/data1/Salim/Underwater/DEIMv2 \
        --cfg  /srv/data1/Salim/Underwater/DEIMv2/configs/deimv2/deimv2_dinov3_l_seaclear.yml \
        --ckpt /srv/data1/Salim/Underwater/DEIMv2/outputs/deimv2_l_seaclear/best_stg2.pth \
        --gt   /srv/data1/Salim/Underwater/DEIMv2/dataset/annotations/instances_val.json \
        --imgs /srv/data1/Salim/Underwater/DEIMv2/dataset/images/val \
        --out  ./preds_seaclear_val.json
"""
import os, sys, json, argparse
import torch, torch.nn as nn
from PIL import Image
import torchvision.transforms as T


def build_model(cfg_path, ckpt_path, device, prefer_ema=True):
    from engine.core.yaml_config import YAMLConfig
    cfg = YAMLConfig(cfg_path, resume=ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if prefer_ema and "ema" in ckpt:
        state = ckpt["ema"]["module"]
    else:
        state = ckpt.get("model") or ckpt.get("ema", {}).get("module")
    cfg.model.load_state_dict(state)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.m = cfg.model.deploy()
            s.p = cfg.postprocessor.deploy()
        def forward(s, x, sz):
            return s.p(s.m(x), sz)

    return M().to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deim-root", default="/srv/data1/Salim/Underwater/DEIMv2",
                    help="added to sys.path so 'engine...' imports resolve")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gt", required=True, help="COCO json for this split (image list/ids)")
    ap.add_argument("--imgs", required=True, help="image dir for this split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score-thr", type=float, default=0.001,
                    help="keep-all default; AP needs the low-score tail")
    ap.add_argument("--resize", type=int, default=640)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N images (0=all)")
    args = ap.parse_args()

    sys.path.insert(0, args.deim_root)
    model = build_model(args.cfg, args.ckpt, args.device, prefer_ema=not args.no_ema)

    tf = T.Compose([T.Resize((args.resize, args.resize)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    gt = json.load(open(args.gt))
    images = gt["images"][:args.limit] if args.limit else gt["images"]
    results = []
    for i, im in enumerate(images):
        path = os.path.join(args.imgs, im["file_name"])
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        pil = Image.open(path).convert("RGB")
        W, H = pil.size
        x = tf(pil).unsqueeze(0).to(args.device)
        sz = torch.tensor([[W, H]]).to(args.device)
        with torch.no_grad():
            labels, boxes, scores = model(x, sz)
        labels, boxes, scores = labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()
        for j in range(len(scores)):
            if scores[j] < args.score_thr:
                continue
            x1, y1, x2, y2 = boxes[j].tolist()
            results.append({
                "image_id": im["id"],
                "category_id": int(labels[j]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(scores[j]),
            })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(images)} images, {len(results)} dets so far")

    json.dump(results, open(args.out, "w"))
    print(f"wrote {len(results)} detections over {len(images)} images -> {args.out}")


if __name__ == "__main__":
    main()