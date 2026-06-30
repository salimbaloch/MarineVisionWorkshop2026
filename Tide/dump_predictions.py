import sys, json, torch, torch.nn as nn
sys.path.insert(0, "/srv/data1/Salim/Underwater/DEIMv2")
from PIL import Image
import torchvision.transforms as T
from engine.core.yaml_config import YAMLConfig

# ---- RUOD-R paths (filled in) ----
CFG   = "/srv/data1/Salim/Underwater/DEIMv2/configs/deimv2/deimv2_dinov3_l_ruodr.yml"
CKPT  = "/srv/data1/Salim/Underwater/DEIMv2/outputs/deimv2_l_ruodr/best_stg2.pth"
GT    = "/srv/data1/Salim/Underwater/RUOD/RUOD-R/instances_test_filtered.json"
IMGS  = "/srv/data1/Salim/Underwater/RUOD/RUOD_pic/test"
OUT   = "/srv/data1/Salim/Underwater/DEIMv2/outputs/deimv2_l_ruodr/preds_ruodr_test.json"
device = "cuda"
# ----------------------------------

cfg = YAMLConfig(CFG, resume=CKPT)
ckpt = torch.load(CKPT, map_location="cpu")
state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
cfg.model.load_state_dict(state)

class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.m = cfg.model.deploy(); s.p = cfg.postprocessor.deploy()
    def forward(s, x, sz): return s.p(s.m(x), sz)
model = M().to(device).eval()

tf = T.Compose([T.Resize((640, 640)), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

gt = json.load(open(GT))
results = []
for im in gt["images"]:
    path = f"{IMGS}/{im['file_name']}"
    pil = Image.open(path).convert("RGB")
    W, H = pil.size
    x = tf(pil).unsqueeze(0).to(device)
    sz = torch.tensor([[W, H]]).to(device)
    with torch.no_grad():
        labels, boxes, scores = model(x, sz)
    labels, boxes, scores = labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()
    for j in range(len(scores)):
        if scores[j] < 0.001: continue
        x1, y1, x2, y2 = boxes[j].tolist()
        results.append({
            "image_id": im["id"],
            "category_id": int(labels[j]),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(scores[j]),
        })
json.dump(results, open(OUT, "w"))
print(f"wrote {len(results)} detections -> {OUT}")