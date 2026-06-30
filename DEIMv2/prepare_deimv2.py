# prepare_deimv2.py
import os, json, random
from collections import defaultdict

ROOT = "/srv/data1/Salim/Underwater/Seaclear_Marine_Debris_Dataset"
OUT  = "/srv/data1/Salim/Underwater/DEIMv2/dataset"
random.seed(42)

coco = json.load(open(os.path.join(ROOT, "dataset.json")))

basename_to_path = {}
for dp, _, files in os.walk(ROOT):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            basename_to_path[f] = os.path.join(dp, f)

imgs = coco["images"][:]
random.shuffle(imgs)
n_val = int(0.10 * len(imgs))
val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]

anns_by_img = defaultdict(list)
for a in coco["annotations"]:
    anns_by_img[a["image_id"]].append(a)

def build(split, split_imgs):
    img_dir = os.path.join(OUT, "images", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(OUT, "annotations"), exist_ok=True)
    out_imgs, out_anns = [], []
    for im in split_imgs:
        src = basename_to_path.get(im["file_name"])
        if src is None:
            continue
        dst = os.path.join(img_dir, im["file_name"])
        if not os.path.exists(dst):
            os.symlink(src, dst)
        out_imgs.append(im)
        out_anns.extend(anns_by_img.get(im["id"], []))
    out = {"images": out_imgs, "annotations": out_anns, "categories": coco["categories"]}
    p = os.path.join(OUT, "annotations", f"instances_{split}.json")
    json.dump(out, open(p, "w"))
    print(f"{split}: {len(out_imgs)} imgs, {len(out_anns)} anns -> {p}")

build("train", train_imgs)
build("val", val_imgs)
print("max category id:", max(c["id"] for c in coco["categories"]), "| #categories:", len(coco["categories"]))