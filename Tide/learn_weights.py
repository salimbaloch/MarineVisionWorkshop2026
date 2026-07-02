
import os, re, glob, json, argparse
import clingo
import optuna

PENALTY_NAMES = {"fit_error", "complexity", "foreign_object", "weak_boundary", "redundancy"}


# --------------------------------------------------------------------- program parts
def load_rules(path):
    
    txt = open(path).read()
    if "%%%%%" in txt:
        txt = txt.split("%%%%%", 1)[1]
    def is_ground_wb(l):
        return bool(re.match(r"\s*(weight|bound)\([a-z_]", l)) and l.rstrip().endswith(").")
    return "\n".join(l for l in txt.splitlines() if not is_ground_wb(l))


def solve(program):
    """Solve to optimum; return (shown_atoms:set[str], cost:list) of the best model."""
    ctl = clingo.Control(["--opt-mode=opt", "--models=0", "--warn=none"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    best = {"atoms": set(), "cost": None}
    def on_model(m):
        best["atoms"] = {str(s) for s in m.symbols(shown=True)}
        best["cost"] = list(m.cost)
    ctl.solve(on_model=on_model)
    return best["atoms"], best["cost"]


def discover_features(rules, sample_facts):
    """Ground rules over sample fact files and collect every feature/penalty name F.
    This is the Optuna search space; it adapts automatically to new classes."""
    feats = set()
    show = "\n#show feature/3.\n#show penalty_feature/3.\n"
    for facts in sample_facts:
        atoms, _ = solve(rules + "\n" + facts + show)
        for a in atoms:
            m = re.match(r"(?:penalty_)?feature\([^,]+,(.+),[^,]+\)$", a)
            if m:
                feats.add(m.group(1))
    return sorted(feats)


# --------------------------------------------------------------------- fact parsing
_BOX = re.compile(r"box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)")

def parse_facts(facts):
    """From a .lp fact string pull: class, {id:box} for regions+primitives, the body
    region id, the set of boundary-primitive ids, and the proposal box."""
    cls = None
    m = re.search(r"class\([^,]+,([a-z_0-9]+)\)", facts)
    if m: cls = m.group(1)
    boxes = {}
    for pat in (r"region_box\((\w+),box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\)",
                r"primitive_box\((\w+),box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\)"):
        for mm in re.finditer(pat, facts):
            boxes[mm.group(1)] = tuple(int(mm.group(i)) for i in range(2, 6))
    body = None
    m = re.search(r"region_role\((\w+),body\)", facts)
    if m: body = m.group(1)
    boundary_ids = set(re.findall(r"boundary_primitive\((\w+)\)", facts))
    prop = None
    m = re.search(r"proposal_box\([^,]+,box\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\)", facts)
    if m: prop = tuple(int(m.group(i)) for i in range(1, 5))
    return dict(cls=cls, boxes=boxes, body=body, boundary_ids=boundary_ids, proposal=prop)


def union_box(bxs):
    xs1 = [b[0] for b in bxs]; ys1 = [b[1] for b in bxs]
    xs2 = [b[2] for b in bxs]; ys2 = [b[3] for b in bxs]
    return (min(xs1), min(ys1), max(xs2), max(ys2))


def refined_box(selected_ids, meta):
    """Box implied by the explanation: union of selected boundary-primitive boxes;
    fallback to the selected body region box; then the proposal box."""
    bb = [meta["boxes"][i] for i in selected_ids
          if i in meta["boundary_ids"] and i in meta["boxes"]]
    if bb:
        return union_box(bb)
    if meta["body"] in selected_ids and meta["body"] in meta["boxes"]:
        return meta["boxes"][meta["body"]]
    return meta["proposal"]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# --------------------------------------------------------------------- weights I/O
def weight_block(cls, wmap, U):
    L = [f"bound({cls},semantic_region,1,{U})."]
    L += [f"weight({cls},{f},{w})." for f, w in sorted(wmap.items())]
    return "\n".join(L) + "\n"


def parse_explains(atoms):
    out = set()
    for a in atoms:
        m = re.match(r"explains\((\w+),\w+\)", a)
        if m: out.add(m.group(1))
    return out


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--lp-dir", required=True)
    ap.add_argument("--gt", required=True, help="gt_boxes.json {stem:[x1,y1,x2,y2]} crop coords")
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--wmin", type=int, default=-5)
    ap.add_argument("--wmax", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rules = load_rules(args.rules)
    gt = {k: tuple(v) for k, v in json.load(open(args.gt)).items()}

    # load training instances of this class
    inst = []
    for p in sorted(glob.glob(os.path.join(args.lp_dir, "*.lp"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in gt:
            continue
        facts = open(p).read()
        meta = parse_facts(facts)
        if meta["cls"] != args.cls:
            continue
        inst.append((stem, facts, meta, gt[stem]))
    if not inst:
        raise SystemExit(f"no .lp for class '{args.cls}' with a GT box in {args.gt}")
    print(f"[{args.cls}] {len(inst)} training detections")

    feats = discover_features(rules, [f for _, f, _, _ in inst[:min(20, len(inst))]])
    print(f"search space: {len(feats)} features + bound U")

    def objective(trial):
        wmap = {}
        for f in feats:
            if f in PENALTY_NAMES:
                wmap[f] = trial.suggest_int(f"w_{f}", 0, args.wmax)      # penalties >= 0
            else:
                wmap[f] = trial.suggest_int(f"w_{f}", args.wmin, args.wmax)
        U = trial.suggest_int("bound_U", 1, 3)
        wb = weight_block(args.cls, wmap, U)
        loss = 0.0
        for stem, facts, meta, gtb in inst:
            atoms, _ = solve(rules + "\n" + facts + "\n" + wb)
            sel = parse_explains(atoms)
            loss += 1.0 - iou(refined_box(sel, meta), gtb)
        return loss / len(inst)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_params
    wmap = {f: best[f"w_{f}"] for f in feats}
    U = best["bound_U"]
    out = args.out or f"weights_{args.cls}.lp"
    with open(out, "w") as fh:
        fh.write(f"% learned by Optuna over {len(inst)} detections; "
                 f"mean(1-IoU)={study.best_value:.4f}\n")
        fh.write(weight_block(args.cls, wmap, U))
    print(f"\nbest mean(1-IoU) = {study.best_value:.4f}  (mean IoU = {1-study.best_value:.4f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()