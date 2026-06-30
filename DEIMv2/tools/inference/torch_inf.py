"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified for SeaClear: class-name labels, adjustable threshold, folder input,
per-image output filenames. (CommonsenseUOD)
"""

import os
import sys
import json
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


# ----------------------------------------------------------------------
# Build id -> name map from the COCO annotation file (so boxes show
# "bottle_plastic 0.87" instead of "3 0.87").
# ----------------------------------------------------------------------
def load_category_names(ann_file):
    if ann_file is None or not os.path.isfile(ann_file):
        return None
    data = json.load(open(ann_file))
    # CocoDetection with remap_mscoco_category=False keeps raw category ids,
    # and the model outputs those same ids. Map id -> name directly.
    return {c["id"]: c["name"] for c in data["categories"]}


# A set of distinct colors so different classes are visually separable.
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#808080", "#ff4500", "#2e8b57", "#daa520", "#00ced1",
    "#9400d3", "#ff1493", "#1e90ff", "#adff2f", "#dc143c", "#00fa9a",
    "#b22222", "#5f9ea0", "#d2691e", "#6a5acd", "#708090", "#ff69b4",
    "#cd5c5c", "#40e0d0", "#ee82ee", "#7fff00",
]


def _color_for(label_id):
    return _PALETTE[int(label_id) % len(_PALETTE)]


def draw(images, labels, boxes, scores, thrh, cat_names, out_path):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, im in enumerate(images):
        drw = ImageDraw.Draw(im)

        scr = scores[i]
        keep = scr > thrh
        lab = labels[i][keep]
        box = boxes[i][keep]
        scrs = scr[keep]

        for j, b in enumerate(box):
            b = list(b)
            lid = int(lab[j].item())
            name = cat_names.get(lid, str(lid)) if cat_names else str(lid)
            color = _color_for(lid)
            text = f"{name} {round(scrs[j].item(), 2)}"

            drw.rectangle(b, outline=color, width=3)

            # text background for readability
            tb = drw.textbbox((b[0], b[1]), text, font=font)
            drw.rectangle([tb[0], tb[1], tb[2], tb[3]], fill=color)
            drw.text((b[0], b[1]), text=text, fill="white", font=font)

        im.save(out_path)
        print(f"  saved -> {out_path}  ({int(keep.sum().item())} detections > {thrh})")


def process_image(model, device, file_path, size, vit_backbone, thrh, cat_names, out_path):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if vit_backbone else T.Lambda(lambda x: x)
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores, thrh, cat_names, out_path)


def process_video(model, device, file_path, size, vit_backbone, thrh, cat_names, out_path):
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if vit_backbone else T.Lambda(lambda x: x)
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # draw onto a temp copy without saving each frame to disk
        scr = scores[0]
        keep = scr > thrh
        drw = ImageDraw.Draw(frame_pil)
        for j, b in enumerate(boxes[0][keep]):
            b = list(b)
            lid = int(labels[0][keep][j].item())
            name = cat_names.get(lid, str(lid)) if cat_names else str(lid)
            color = _color_for(lid)
            drw.rectangle(b, outline=color, width=3)
            drw.text((b[0], b[1]), f"{name} {round(scr[keep][j].item(),2)}", fill="white")

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        out.write(frame)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Video processing complete. Saved -> {out_path}")


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    device = args.device
    model = Model().to(device)
    model.eval()
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = bool(cfg.yaml_cfg.get('DINOv3STAs', False))

    cat_names = load_category_names(args.ann_file)
    if cat_names:
        print(f"Loaded {len(cat_names)} class names from {args.ann_file}")
    else:
        print("No annotation file given (or not found): showing numeric class ids.")

    os.makedirs(args.output_dir, exist_ok=True)

    # gather inputs: a single file or every image in a folder
    if os.path.isdir(args.input):
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        files = sorted([f for e in exts for f in glob.glob(os.path.join(args.input, e))])
    else:
        files = [args.input]
    print(f"{len(files)} input file(s).")

    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        ext = os.path.splitext(fp)[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            out_path = os.path.join(args.output_dir, f"{stem}_det.jpg")
            process_image(model, device, fp, img_size, vit_backbone,
                          args.threshold, cat_names, out_path)
        else:
            out_path = os.path.join(args.output_dir, f"{stem}_det.mp4")
            process_video(model, device, fp, img_size, vit_backbone,
                          args.threshold, cat_names, out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='image, video, or a folder of images')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-t', '--threshold', type=float, default=0.45,
                        help='confidence threshold for drawing boxes')
    parser.add_argument('-a', '--ann_file', type=str, default=None,
                        help='COCO json (e.g. dataset/annotations/instances_val.json) for class names')
    parser.add_argument('-o', '--output_dir', type=str, default='vis_results',
                        help='where to save annotated images/videos')
    args = parser.parse_args()
    main(args)