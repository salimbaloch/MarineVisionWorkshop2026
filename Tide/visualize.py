import sys, json, os, random
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

GT, PRED, IMGS, OUTDIR = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 8
SCORE_TH = 0.30
IOU_HIT = 0.50
random.seed(0)
os.makedirs(OUTDIR, exist_ok=True)

def iou(b1, b2):
    x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
    xa,ya=max(x1,x2),max(y1,y2); xb,yb=min(x1+w1,x2+w2),min(y1+h1,y2+h2)
    inter=max(0,xb-xa)*max(0,yb-ya)
    return inter/(w1*h1+w2*h2-inter+1e-9)

gt = json.load(open(GT))
cats = {c["id"]: c["name"] for c in gt["categories"]}
imgs = {im["id"]: im for im in gt["images"]}
gt_by = defaultdict(list); pred_by = defaultdict(list)
for a in gt["annotations"]: gt_by[a["image_id"]].append(a)
for p in json.load(open(PRED)):
    if p["score"] >= SCORE_TH: pred_by[p["image_id"]].append(p)

pick = random.sample([i for i in gt_by if gt_by[i]], min(N, len(gt_by)))
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except: font = ImageFont.load_default()

def box(d, bb, c, t):
    x,y,w,h=bb; d.rectangle([x,y,x+w,y+h], outline=c, width=3)
    if t: d.text((x+2, max(0,y-16)), t, fill=c, font=font)

for img_id in pick:
    info = imgs[img_id]; path = os.path.join(IMGS, info["file_name"])
    if not os.path.exists(path): print("missing", path); continue
    img = Image.open(path).convert("RGB"); d = ImageDraw.Draw(img)
    gts = gt_by[img_id]; preds = sorted(pred_by[img_id], key=lambda x:-x["score"])

    used = [False]*len(gts); matched = 0; fp = 0
    for p in preds:
        bi,bj=0,-1
        for j,g in enumerate(gts):
            if used[j]: continue
            i = iou(p["bbox"], g["bbox"])
            if i>bi: bi,bj=i,j
        if bi>=IOU_HIT and bj>=0: used[bj]=True; matched+=1
        else: fp+=1
    miss = sum(1 for u in used if not u)

    for a in gts: box(d, a["bbox"], (0,220,0), cats[a["category_id"]])
    for p in preds: box(d, p["bbox"], (255,40,40), f"{cats.get(p['category_id'],'?')} {p['score']:.2f}")

    hdr = f"GT {len(gts)} | Pred {len(preds)} | matched {matched} | FP {fp} | miss {miss}"
    d.rectangle([0,0,len(hdr)*10+16,26], fill=(0,0,0))
    d.text((8,5), hdr, fill=(255,255,0), font=font)
    out = os.path.join(OUTDIR, f"{os.path.splitext(info['file_name'])[0]}_vis.jpg")
    img.save(out, quality=90)
    print(f"{info['file_name']}: {hdr}")

print("\nGREEN=GT  RED=pred(score).  matched=correct@IoU0.5  FP=red w/o GT  miss=green w/o red")