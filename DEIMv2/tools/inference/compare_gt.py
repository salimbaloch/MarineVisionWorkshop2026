"""
GT-vs-Prediction comparison for SeaClear / DEIMv2 (CommonsenseUOD).
Draws ground-truth boxes (green) and model predictions (red) side by side,
so you can see true positives, missed objects, and false positives at a glance.

Built on top of DEIMv2's torch_inf.py model-loading path.
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


GT_COLOR = "#00e000"     # green  = ground truth
PRED_COLOR = "#ff2020"   # red    = prediction


def get_font(sz=15):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def load_coco(ann_file):
    data = json.load(open(ann_file))
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    # group GT boxes per image filename
    imgid2info = {im["id"]: im for im in data["images"]}
    gt_by_file = defaultdict(list)
    for a in data["annotations"]:
        im = imgid2info[a["image_id"]]
        gt_by_file[im["file_name"]].append((a["bbox"], a["category_id"]))  # bbox = [x,y,w,h]
    return id2name, gt_by_file


def draw_boxes_xywh(im, anns, id2name, color):
    """anns: list of ([x,y,w,h], cat_id). COCO format."""
    drw = ImageDraw.Draw(im)
    font = get_font()
    for (x, y, w, h), cid in anns:
        x2, y2 = x + w, y + h
        name = id2name.get(int(cid), str(cid))
        drw.rectangle([x, y, x2, y2], outline=color, width=3)
        tb = drw.textbbox((x, y), name, font=font)
        drw.rectangle([tb[0], tb[1], tb[2], tb[3]], fill=color)
        drw.text((x, y), name, fill="white", font=font)


def draw_boxes_xyxy(im, labels, boxes, scores, thrh, id2name, color):
    """boxes: xyxy from postprocessor (already in original-image scale)."""
    drw = ImageDraw.Draw(im)
    font = get_font()
    keep = scores > thrh
    for b, lid, sc in zip(boxes[keep], labels[keep], scores[keep]):
        b = [float(v) for v in b]
        name = id2name.get(int(lid), str(int(lid)))
        text = f"{name} {round(float(sc), 2)}"
        drw.rectangle(b, outline=color, width=3)
        tb = drw.textbbox((b[0], b[1]), text, font=font)
        drw.rectangle([tb[0], tb[1], tb[2], tb[3]], fill=color)
        drw.text((b[0], b[1]), text, fill="white", font=font)


def build_model(cfg, resume, device):
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    checkpoint = torch.load(resume, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    m = Model().to(device)
    m.eval()
    return m


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    device = args.device
    model = build_model(cfg, args.resume, device)
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = bool(cfg.yaml_cfg.get('DINOv3STAs', False))

    id2name, gt_by_file = load_coco(args.ann_file)
    print(f"Loaded {len(id2name)} classes; GT for {len(gt_by_file)} images.")

    os.makedirs(args.output_dir, exist_ok=True)

    transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if vit_backbone else T.Lambda(lambda x: x)
    ])

    # collect input images
    if os.path.isdir(args.input):
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        files = sorted([f for e in exts for f in glob.glob(os.path.join(args.input, e))])
    else:
        files = [args.input]
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} image(s) to process.")

    for fp in files:
        fname = os.path.basename(fp)
        im_pil = Image.open(fp).convert('RGB')
        w, h = im_pil.size

        # --- prediction ---
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(im_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            labels, boxes, scores = model(im_data, orig_size)
        labels, boxes, scores = labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()

        # --- two copies: left = GT, right = prediction ---
        left = im_pil.copy()
        right = im_pil.copy()

        gt_anns = gt_by_file.get(fname, [])
        draw_boxes_xywh(left, gt_anns, id2name, GT_COLOR)
        draw_boxes_xyxy(right, labels, boxes, scores, args.threshold, id2name, PRED_COLOR)

        # stitch side by side with labels
        gap = 8
        canvas = Image.new("RGB", (w * 2 + gap, h + 30), (20, 20, 20))
        canvas.paste(left, (0, 30))
        canvas.paste(right, (w + gap, 30))
        d = ImageDraw.Draw(canvas)
        f = get_font(18)
        d.text((10, 6), f"GROUND TRUTH ({len(gt_anns)} boxes)", fill=GT_COLOR, font=f)
        n_pred = int((scores > args.threshold).sum().item())
        d.text((w + gap + 10, 6), f"PREDICTION ({n_pred} boxes > {args.threshold})",
               fill=PRED_COLOR, font=f)

        out_path = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_cmp.jpg")
        canvas.save(out_path)
        print(f"  {fname}: GT={len(gt_anns)}  pred={n_pred}  -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='image or folder of images (use val folder)')
    parser.add_argument('-a', '--ann_file', type=str, required=True,
                        help='COCO json matching the images, e.g. instances_val.json')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-t', '--threshold', type=float, default=0.4)
    parser.add_argument('-o', '--output_dir', type=str, default='cmp_results')
    parser.add_argument('-l', '--limit', type=int, default=0,
                        help='process only the first N images (0 = all)')
    args = parser.parse_args()
    main(args)