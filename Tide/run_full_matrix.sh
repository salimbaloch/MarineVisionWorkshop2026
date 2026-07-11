#!/usr/bin/env bash
# =============================================================================
# run_full_matrix.sh -- FULL paper evaluation across SeaClear / DUO / RUOD-R.
#
# For every listed high-support class (profile mapped to closest structural
# match), per DATASET:
#   decompose val/test (--dump-emb) -> add_overlap -> dump_support -> make_labels
#   -> faithfulness (grounding/discriminativeness)          [all classes]
#   -> regressor refined boxes (eval_coco_ap, learned alpha) [all classes]
#   -> selector refined boxes (build_candidates+train_selector) [all classes]
#   -> per-class baseline vs refined AP rows
# Then, per dataset, EXACT all-class mAP: swap the decomposed classes' refined
# boxes into the FULL predictions and COCO-eval over ALL classes (undecomposed
# classes keep detector boxes -> unchanged -> cancel in the delta -> mAP exact).
#
# One refiner (SeaClear-urchin weights+regressor) applied across all classes:
# honest, expected ~baseline everywhere (the paper's point). NOT per-class tuned.
#
# Profiles are approximate for non-urchin/fish/bottle classes (they only need to
# emit a candidate set); mismatched-profile grounding will be poor -- honest data.
#
# Robustness: each class is wrapped; a failure logs to $OUT/FAILED.log and the
# matrix continues. Re-running skips finished decompositions.
#
# Est: ~3-4h, mostly unattended. RUOD fish (24k) is the biggest single decompose.
#
# Usage:
#   ./run_full_matrix.sh                 # everything
#   ./run_full_matrix.sh seaclear        # one dataset
#   ./run_full_matrix.sh ruodr fish      # one dataset+class
# =============================================================================
set -uo pipefail                      # NOT -e: we want to survive per-class fails

TIDE=/srv/data1/Salim/Underwater/Tide
ASP=/srv/data1/Salim/Underwater/object_explanation_v2.lp
OUT=/srv/data1/Salim/Underwater/matrix_out
PIDINET=/srv/data1/Salim/Underwater/table7_pidinet.pth
# the single refiner applied everywhere (trained on SeaClear urchin)
REF_WEIGHTS=/srv/data1/Salim/Underwater/stage_out/urchin/alt_animal_urchin_weights.lp
REF_CKPT=/srv/data1/Salim/Underwater/stage_out/urchin/alt_animal_urchin_regressor.pt
mkdir -p "$OUT"; : > "$OUT/FAILED.log"

# ---- per-dataset repo / cfg / gt / img / preds -----------------------------
declare -A REPO CFG CKPT GT IMG PRED
REPO[seaclear]=/srv/data1/Salim/Underwater/DEIMv2
GT[seaclear]=/srv/data1/Salim/Underwater/DEIMv2/dataset/annotations/instances_val.json
IMG[seaclear]=/srv/data1/Salim/Underwater/DEIMv2/dataset/images/val
CKPT[seaclear]=/srv/data1/Salim/Underwater/DEIMv2/outputs/deimv2_dinov3_l_coco/best_stg2.pth
PRED[seaclear]=/srv/data1/Salim/Underwater/DEIMv2/outputs/deimv2_dinov3_l_coco/predictions_val.json

REPO[duo]=/srv/data1/Salim/Underwater/DEIMv2
GT[duo]=/srv/data1/Salim/Underwater/DUO/DUO/annotations/instances_test.json
IMG[duo]=/srv/data1/Salim/Underwater/DUO/DUO/images/test
CKPT[duo]=/srv/data1/Salim/Underwater/DEIMv2/outputs/deimv2_l_duo/best_stg2.pth
PRED[duo]=/srv/data1/Salim/Underwater/DEIMv2/outputs/deimv2_l_duo/predictions_test.json

REPO[ruodr]=/srv/data1/Salim/Underwater/DEIMv2
GT[ruodr]=/srv/data1/Salim/Underwater/RUOD/RUOD-R/instances_test.json
IMG[ruodr]=/srv/data1/Salim/Underwater/RUOD/RUOD_pic/test
CKPT[ruodr]=/srv/data1/Salim/Underwater/DEIMv2/outputs/deimv2_l_ruodr/best_stg2.pth
PRED[ruodr]=/srv/data1/Salim/Underwater/DEIMv2/outputs/deimv2_l_ruodr/preds_ruodr_test.json

# ---- class lists: "name:profile:scorethr" (thr blank -> default 0.2) --------
declare -A CLASSES
CLASSES[seaclear]="animal_urchin:urchin: animal_fish:fish: bottle_plastic:bottle: \
animal_etc:urchin: tire_rubber:urchin: bottle_glass:bottle: rope_fiber:fish: \
tube_cement:urchin: animal_shells:urchin: animal_sponge:urchin:"
CLASSES[duo]="echinus:urchin: starfish:urchin: holothurian:fish: scallop:urchin:"
CLASSES[ruodr]="fish:fish:0.3 corals:urchin: echinus:urchin: jellyfish:fish: \
scallop:urchin: holothurian:fish: starfish:urchin:"

DEF_THR=0.2
DATASETS=(seaclear duo ruodr)
[ "$#" -ge 1 ] && DATASETS=("$1")
ONLY_CLASS="${2:-}"

cd "$TIDE"
catid () { python3 -c "import json,sys;print(next((c['id'] for c in json.load(open(sys.argv[1]))['categories'] if c['name']==sys.argv[2]), -1))" "$1" "$2"; }

fail () { echo "[FAIL] $1" | tee -a "$OUT/FAILED.log"; }

for DS in "${DATASETS[@]}"; do
  R=${REPO[$DS]}; GTF=${GT[$DS]}; IMGD=${IMG[$DS]}; W=${CKPT[$DS]}; PREDF=${PRED[$DS]}
  echo ""; echo "################## DATASET $DS ##################"
  [ -f "$PREDF" ] || { fail "$DS: predictions missing ($PREDF)"; continue; }
  [ -f "$GTF" ]   || { fail "$DS: GT missing ($GTF)"; continue; }
  DSOUT=$OUT/$DS; mkdir -p "$DSOUT"

  # weight block for solve-based faithfulness (from rules; no Optuna needed here)
  WB=$DSOUT/wb.lp
  python3 - "$ASP" "$WB" <<'PY'
import re,sys
wb=[l for l in open(sys.argv[1]).read().splitlines()
    if re.match(r"\s*(weight|bound)\([a-z_]",l) and l.rstrip().endswith(").")]
open(sys.argv[2],"w").write("\n".join(wb)+"\n")
PY

  DECODED_CLASSES=()        # track which classes we successfully refined, for mAP swap
  for SPEC in ${CLASSES[$DS]}; do
    CN="${SPEC%%:*}"; REST="${SPEC#*:}"; PROF="${REST%%:*}"; THR="${REST#*:}"
    [ -z "$THR" ] && THR=$DEF_THR
    [ -n "$ONLY_CLASS" ] && [ "$CN" != "$ONLY_CLASS" ] && continue
    CID=$(catid "$GTF" "$CN")
    [ "$CID" = "-1" ] && { fail "$DS/$CN: not in GT"; continue; }
    C=$DSOUT/$CN; DEC=$C/dec; SUP=$DEC/support; mkdir -p "$C"
    echo ""; echo "===== $DS / $CN  (profile=$PROF, cat=$CID, thr=$THR) ====="

    # 1) decompose (with embeddings for the regressor)
    if [ ! -f "$DEC/crop_meta.json" ]; then
      python m4decompose.py --profile "$PROF" --class-name "$CN" \
        --repo-root "$R" --weights "$W" --pidinet-weights "$PIDINET" \
        --pred-json "$PREDF" --gt-json "$GTF" --img-root "$IMGD" \
        --box-format xywh --score-thr "$THR" \
        --dump-emb --emb-dir "$DEC/emb" --out-dir "$DEC" \
        --max-dets 100000 --fresh 2>>"$OUT/FAILED.log" \
        || { fail "$DS/$CN: decompose"; continue; }
    fi
    python add_overlap_facts.py --lp-dir "$DEC" >/dev/null 2>&1 || true

    # 2) support maps (for faithfulness/localize candidates)
    if [ ! -d "$SUP" ]; then
      python dump_support.py --class-name "$CN" --profile "$PROF" \
        --repo-root "$R" --weights "$W" \
        --pred-json "$PREDF" --gt-json "$GTF" --img-root "$IMGD" \
        --box-format xywh --score-thr "$THR" --out-dir "$SUP" \
        2>>"$OUT/FAILED.log" || fail "$DS/$CN: dump_support (non-fatal)"
    fi

    # 3) COCO-exact labels
    python make_labels.py --preds "$PREDF" --gt "$GTF" \
      --class-name "$CN" --crop-meta "$DEC/crop_meta.json" \
      --out "$C/labels.json" 2>>"$OUT/FAILED.log" \
      || { fail "$DS/$CN: make_labels"; continue; }

    # 4) faithfulness (grounding/discriminativeness; solve-based selectivity)
    python faithfulness.py --lp-dir "$DEC" --labels "$C/labels.json" \
      --gt-boxes "$DEC/gt_boxes.json" --class "$CN" --dataset "$DS" \
      --out "$C/faith_val.json" \
      >"$C/faith.txt" 2>>"$OUT/FAILED.log" || fail "$DS/$CN: faithfulness (non-fatal)"

    # 5) REGRESSOR refined boxes (learned alpha) -> writes eval_reg_report.json + refined preds
    python eval_coco_ap.py --rules "$ASP" --weights "$REF_WEIGHTS" --ckpt "$REF_CKPT" \
      --lp-dir "$DEC" --emb-dir "$DEC/emb" --crop-meta "$DEC/crop_meta.json" \
      --preds "$PREDF" --coco-gt "$GTF" --class "$CN" --cat-id "$CID" --solve-timeout 5 \
      --out "$C/refined_reg.json" >"$C/reg.txt" 2>>"$OUT/FAILED.log" \
      || fail "$DS/$CN: regressor eval (non-fatal)"

    # 6) SELECTOR refined boxes (candidates -> train on THIS class -> gated swap)
    #    (per-class selector is cheap and fairer than cross-applying urchin's.)
    python build_candidates.py --lp-dir "$DEC" \
      --crop-meta "$DEC/crop_meta.json" --gt-boxes "$DEC/gt_boxes.json" \
      --class "$CN" ${SUP:+--support-dir "$SUP"} \
      \
      --out "$C/cand.npz" 2>>"$OUT/FAILED.log" \
      && python train_selector.py \
           --train-cand "$C/cand.npz" --val-cand "$C/cand.npz" \
           --val-crop-meta "$DEC/crop_meta.json" \
           --preds "$PREDF" --coco-gt "$GTF" --class "$CN" --cat-id "$CID" \
           --out "$C/refined_sel.json" >"$C/sel.txt" 2>>"$OUT/FAILED.log" \
      || fail "$DS/$CN: selector (non-fatal)"

    DECODED_CLASSES+=("$CN:$CID:$C")
    echo "  done $DS/$CN"
  done

  # ---- EXACT all-class mAP: swap refined boxes for decoded classes into full preds ----
  if [ ${#DECODED_CLASSES[@]} -gt 0 ]; then
    echo ""; echo "=== $DS: exact all-class mAP (baseline vs regressor vs selector) ==="
    python3 - "$PREDF" "$GTF" "$DSOUT/mAP.json" "${DECODED_CLASSES[@]}" <<'PY'
import io, json, sys, contextlib
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

preds_path, gt_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
decoded = sys.argv[4:]   # "name:cid:dir"

def mAP(preds):
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO(gt_path); d = g.loadRes([dict(p) for p in preds])
        E = COCOeval(g, d, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    return {"AP": E.stats[0], "AP50": E.stats[1], "AP75": E.stats[2]}

def key(catid, img, x, y, w, h):
    return (int(catid), int(img), round(x,2), round(y,2), round(x+w,2), round(y+h,2))

base = json.load(open(preds_path))
# build swap maps from each decoded class's refined-preds jsons
# Cleaner: each refined_*.json is the full pred list with ONE class refined. Merge by taking,
# for each decoded class, that file's rows for that class; for all else, baseline rows.
def merge(fname):
    # start from baseline; for each decoded class, replace its rows with the refined file's rows
    by_class_refined = {}
    for spec in decoded:
        cn, cid, cdir = spec.split(":"); cid=int(cid)
        try:
            rp = json.load(open(f"{cdir}/{fname}"))
        except Exception:
            continue
        by_class_refined[cid] = [d for d in rp if d.get("category_id")==cid]
    out = []
    for d in base:
        c = d.get("category_id")
        if c in by_class_refined:
            continue  # will add refined rows below
        out.append(d)
    for cid, rows in by_class_refined.items():
        out.extend(rows)
    return out

res = {"baseline": mAP(base)}
reg = merge("refined_reg.json"); res["regressor"] = mAP(reg)
sel = merge("refined_sel.json"); res["selector"] = mAP(sel)
res["n_decoded_classes"] = len(decoded)
json.dump(res, open(out_path,"w"), indent=1)
print(f"  all-class mAP over full class set (decoded {len(decoded)} classes):")
for k in ("baseline","regressor","selector"):
    m=res[k]; print(f"    {k:10s}  AP {m['AP']:.4f}  AP50 {m['AP50']:.4f}  AP75 {m['AP75']:.4f}")
print(f"  wrote {out_path}")
PY
  fi
done

# ---- compile per-class table across everything ----
echo ""; echo "=== compile per-class table ==="
python compile_table.py --root "$OUT" --out-md "$OUT/results.md" --out-tex "$OUT/results.tex" 2>/dev/null || \
  echo "(compile_table: point it at faith_val.json naming if needed)"

echo ""
echo "DONE. Artifacts:"
echo "  per-dataset mAP : $OUT/<ds>/mAP.json   (baseline vs regressor vs selector, exact)"
echo "  per-class faith : $OUT/<ds>/<class>/faith_val.json"
echo "  per-class AP    : $OUT/<ds>/<class>/{reg,sel}.txt"
echo "  failures        : $OUT/FAILED.log"