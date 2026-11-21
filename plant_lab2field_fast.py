# %% [markdown]
# # Lab-to-Field Plant Disease Classification: FAST version (about 1 hour)
# Model: EfficientNet-B0 | Conditions: C0 (baseline) vs C3 (background replacement + photometric + erasing)
# Outputs: 5-fold cross-validation table, accuracy/loss curves, confusion matrices, Grad-CAM interpretability.
#
# **Setup:** Settings -> GPU T4 x1 and Internet ON. Add Data: `abdallahalidev/plantvillage-dataset`
# and `nirmalsankalana/plantdoc-dataset`.
# **Run:** keep `SMOKE = True` first (about 5 min), then set `SMOKE = False` and run all cells (about 45-60 min).

# %% Step 1: Install and imports
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "grad-cam"], check=False)

import os, re, json, random, time, zipfile, warnings
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
import matplotlib.pyplot as plt
from scipy.stats import binomtest, wilcoxon, ttest_rel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
warnings.filterwarnings("ignore")

T0 = time.time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
print("device:", DEVICE, "| torch", torch.__version__, "| timm", timm.__version__)

# %% Step 2: Configuration
SMOKE = True            # True: 2 folds, 1 epoch, 60 images/class (about 5 min). False: real run.
PRETRAINED = True       # needs Internet ON

CFG = dict(model="efficientnet_b0", conds=["C0", "C3"], folds=5, epochs=5, bs=64, img=224,
           cap_per_class=250, lr_head=1e-3, lr_backbone=3e-4, wd=1e-4, workers=4, seed=42)
if SMOKE:
    CFG.update(folds=2, epochs=1, cap_per_class=60, workers=2)

OUT = Path("/kaggle/working") / ("fast_smoke" if SMOKE else "fast")
for d in ["preds", "ckpt", "figs", "history"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

SIG = {k: CFG[k] for k in ["folds", "epochs", "bs", "img", "cap_per_class", "lr_head", "lr_backbone", "wd", "seed", "model"]}
SIG_FILE = OUT / "config_signature.json"
if SIG_FILE.exists():
    old = json.load(open(SIG_FILE))
    assert old == SIG, f"Settings changed since the saved results.\nsaved: {old}\nnow:   {SIG}\nDelete {OUT} to start fresh."
else:
    json.dump(SIG, open(SIG_FILE, "w"))

PV_COLOR, PV_SEG, BG_ROOT = None, None, None
PD_ROOT = Path("/kaggle/input/datasets/nirmalsankalana/plantdoc-dataset")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
EXT = {".jpg", ".jpeg", ".png"}
def list_images(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in EXT]

# %% Step 3: Find the datasets
def walk_find(root, pred, maxdepth=5):
    hits = []
    for dp, dn, fn in os.walk(root):
        depth = len(Path(dp).relative_to(root).parts)
        if depth >= maxdepth: dn[:] = []
        if pred(Path(dp)): hits.append(Path(dp))
    return hits

ROOT = Path("/kaggle/input")
if PV_COLOR is None:
    hits = walk_find(ROOT, lambda p: p.name.lower() == "color")
    PV_COLOR = hits[0] if hits else None
if PV_SEG is None and PV_COLOR is not None and (PV_COLOR.parent / "segmented").exists():
    PV_SEG = PV_COLOR.parent / "segmented"
if PD_ROOT is None or not Path(PD_ROOT).exists():
    hits = walk_find(ROOT, lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                     and "plantvillage" not in str(p).lower())
    PD_ROOT = hits[0] if hits else None
if BG_ROOT is None:
    hits = walk_find(ROOT, lambda p: "intelimageclassification" in norm(p.name))
    BG_ROOT = hits[0] if hits else None

print("PlantVillage color :", PV_COLOR)
print("PlantVillage segm. :", PV_SEG)
print("PlantDoc root      :", PD_ROOT)
print("Backgrounds        :", BG_ROOT, "(None -> synthetic backgrounds)")
assert PV_COLOR is not None, "PlantVillage not found: add abdallahalidev/plantvillage-dataset"
assert PD_ROOT is not None, "PlantDoc not found: add the PlantDoc dataset or set PD_ROOT"

# %% Step 4: Load image lists and map PlantDoc classes to PlantVillage classes
seg_idx = {}
if PV_SEG is not None:
    for p in list_images(PV_SEG):
        seg_idx[(p.parent.name, re.sub(r"_final_masked$", "", p.stem.lower()))] = str(p)

df_pv = pd.DataFrame(
    [dict(path=str(p), cls=p.parent.name, seg=seg_idx.get((p.parent.name, p.stem.lower())))
     for p in list_images(PV_COLOR)])
df_pd = pd.DataFrame(
    [dict(path=str(p), raw=p.parent.name) for p in list_images(PD_ROOT)
     if p.parent.name.lower() not in ("train", "test", "val")])
print("PlantVillage:", len(df_pv), "images,", df_pv.cls.nunique(), "classes | PlantDoc:", len(df_pd), "images")

PD2PV = {
    "Apple Scab Leaf": "Apple___Apple_scab", "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust", "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy", "Potato leaf": "Potato___healthy",
    "Potato leaf early blight": "Potato___Early_blight", "Potato leaf late blight": "Potato___Late_blight",
    "Strawberry leaf": "Strawberry___healthy", "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot", "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot", "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold", "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy", "squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Blueberry leaf": "Blueberry___healthy", "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Raspberry leaf": "Raspberry___healthy", "Soyabean leaf": "Soybean___healthy",
}
pv_by_norm = {norm(c): c for c in df_pv.cls.unique()}
map_norm = {norm(k): norm(v) for k, v in PD2PV.items()}
df_pd["pv_cls"] = df_pd.raw.map(lambda r: pv_by_norm.get(map_norm.get(norm(r))))
print("PlantDoc folders NOT mapped:", sorted(df_pd[df_pd.pv_cls.isna()].raw.unique()))

cnt = df_pd.dropna(subset=["pv_cls"]).groupby("pv_cls").size()
CLASSES = sorted(cnt[cnt >= 10].index)
cls2idx = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
short = [c.replace("_(maize)", "").replace("_(including_sour)", "").replace("___", " | ").replace("_", " ")
         for c in CLASSES]
print(f"Shared classes: {NUM_CLASSES}")
df_pv = df_pv[df_pv.cls.isin(CLASSES)].copy(); df_pv["y"] = df_pv.cls.map(cls2idx)
df_fd = df_pd[df_pd.pv_cls.isin(CLASSES)].copy(); df_fd["y"] = df_fd.pv_cls.map(cls2idx)
pd.DataFrame({"index": range(NUM_CLASSES), "class": CLASSES, "field_images": cnt[CLASSES].values}
             ).to_csv(OUT / "class_index.csv", index=False)

# %% Step 5: Cross-validation folds (PlantVillage pool) and the field test set (PlantDoc)
df_pool = pd.concat([g.sample(min(len(g), CFG["cap_per_class"]), random_state=0)
                     for _, g in df_pv.groupby("cls")]).reset_index(drop=True)
skf = StratifiedKFold(n_splits=CFG["folds"], shuffle=True, random_state=CFG["seed"])
FOLDS = list(skf.split(df_pool, df_pool.y))
# Same field split as the full study: 30% field_val (unused here) and 70% field_test (final reporting only)
df_fv, df_ft = train_test_split(df_fd, test_size=0.7, stratify=df_fd.y, random_state=42)
print(f"PlantVillage pool: {len(df_pool)} | folds: {CFG['folds']} | field test: {len(df_ft)}")

# %% Step 6: Background replacement module (eroded mask, soft edge, earthy synthetic backgrounds)
def hsv_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (10, 25, 25), (100, 255, 255))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0

def leaf_mask(rgb, seg_path=None):
    if isinstance(seg_path, str):
        s = cv2.imread(seg_path)
        if s is not None:
            if s.shape[:2] != rgb.shape[:2]:
                s = cv2.resize(s, (rgb.shape[1], rgb.shape[0]))
            m = (cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) > 10).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
            return m > 0
    return hsv_mask(rgb)

BG_PATHS = []
if BG_ROOT is not None:
    BG_PATHS = [str(p) for p in list_images(BG_ROOT)]
    random.Random(0).shuffle(BG_PATHS); BG_PATHS = BG_PATHS[:1500]
print("real background images:", len(BG_PATHS))

EARTH = np.array([[34,70,30],[60,100,40],[90,70,40],[120,95,60],
                  [150,140,120],[110,110,105],[200,200,195],[30,45,25]])
def synthetic_bg(h, w):
    rng = np.random.default_rng(random.getrandbits(32))
    pick = lambda: np.clip(EARTH[rng.integers(0, len(EARTH))] + rng.integers(-20, 20, 3), 0, 255)
    c1, c2 = pick(), pick()
    t = np.linspace(0, 1, h)[:, None, None] if rng.random() < .5 else np.linspace(0, 1, w)[None, :, None]
    bg = np.clip(np.zeros((h, w, 3)) + c1 * (1 - t) + c2 * t + rng.normal(0, 12, (h, w, 3)), 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(4, 10))):
        cv2.ellipse(bg, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                    (int(rng.integers(10, w // 3)), int(rng.integers(10, h // 3))),
                    float(rng.integers(0, 180)), 0, 360, tuple(int(v) for v in pick()), -1)
    return cv2.GaussianBlur(bg, (0, 0), 3)

def random_bg(h, w):
    if BG_PATHS:
        b = cv2.imread(random.choice(BG_PATHS))
        if b is not None:
            return cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (w, h))
    return synthetic_bg(h, w)

def replace_bg(pil, seg_path=None, p=0.5):
    if random.random() > p: return pil
    rgb = np.array(pil); h, w = rgb.shape[:2]
    m = leaf_mask(rgb, seg_path)
    if m.mean() < 0.05 or m.mean() > 0.98: return pil
    alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[..., None]
    return Image.fromarray(np.clip(alpha * rgb + (1 - alpha) * random_bg(h, w), 0, 255).astype(np.uint8))

fig, ax = plt.subplots(3, 8, figsize=(20, 8))
for j, (_, r) in enumerate(df_pool.sample(8, random_state=1).iterrows()):
    im = Image.open(r.path).convert("RGB")
    ax[0, j].imshow(im); ax[1, j].imshow(leaf_mask(np.array(im), r.seg), cmap="gray")
    ax[2, j].imshow(replace_bg(im, r.seg, p=1.0))
    for a in ax[:, j]: a.axis("off")
ax[0, 0].set_title("original"); ax[1, 0].set_title("mask"); ax[2, 0].set_title("new background")
plt.tight_layout(); plt.savefig(OUT / "figs/bg_check.png", dpi=120); plt.show()

fig, ax = plt.subplots(2, 6, figsize=(15, 5.5))
for j, (_, r) in enumerate(df_pool.sample(6, random_state=2).iterrows()):
    ax[0, j].imshow(Image.open(r.path)); ax[0, j].axis("off")
for j, (_, r) in enumerate(df_ft.sample(6, random_state=2).iterrows()):
    ax[1, j].imshow(Image.open(r.path)); ax[1, j].axis("off")
ax[0, 0].set_title("PlantVillage (lab)"); ax[1, 0].set_title("PlantDoc (field)")
plt.tight_layout(); plt.savefig(OUT / "figs/lab_vs_field.png", dpi=150); plt.show()

# %% Step 7: Datasets and transforms
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
EVAL_TF = T.Compose([T.ToTensor(), NORM])
BG_P = {"C0": 0.0, "C1": 0.0, "C2": 0.5, "C3": 0.5}

def build_tf(cond):
    ops = [T.Resize((CFG["img"], CFG["img"])), T.RandomHorizontalFlip(), T.RandomRotation(20)]
    if cond in ("C1", "C3"):
        ops += [T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2)]
    ops += [T.ToTensor(), NORM]
    if cond == "C3":
        ops += [T.RandomErasing(p=0.3, scale=(0.02, 0.15))]
    return T.Compose(ops)

class TrainDS(Dataset):
    def __init__(self, df, tf, bg_p):
        self.rows, self.tf, self.bg_p = df.to_dict("records"), tf, bg_p
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        if self.bg_p > 0: img = replace_bg(img, r["seg"], self.bg_p)
        return self.tf(img), int(r["y"])

class CachedDS(Dataset):
    def __init__(self, df, size=224):
        self.imgs = [Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR) for p in df.path]
        self.y = df.y.values
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return EVAL_TF(self.imgs[i]), int(self.y[i])

VAL_DS = [CachedDS(df_pool.iloc[va]) for _, va in FOLDS]
field_test_ds = CachedDS(df_ft)
np.save(OUT / "preds/field_labels.npy", field_test_ds.y)
print("cached:", [len(v) for v in VAL_DS], "validation folds |", len(field_test_ds), "field test images")

# %% Step 8: Training function with per-epoch history
def make_model(name, n=None, pretrained=PRETRAINED):
    return timm.create_model(name, pretrained=pretrained, num_classes=n or NUM_CLASSES)

@torch.no_grad()
def evaluate(model, ds):
    model.eval(); loss_sum, P = 0.0, []
    for x, y in DataLoader(ds, batch_size=256, shuffle=False, num_workers=0):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP):
            out = model(x)
        loss_sum += F.cross_entropy(out.float(), y, reduction="sum").item()
        P.append(out.argmax(1).cpu())
    P = torch.cat(P).numpy()
    return loss_sum / len(ds), float((P == ds.y).mean()), P

def run_fold(cond, k):
    t0 = time.time(); tr_i, _ = FOLDS[k]
    set_seed(CFG["seed"] + k)
    g = torch.Generator(); g.manual_seed(CFG["seed"] + k)
    dl = DataLoader(TrainDS(df_pool.iloc[tr_i], build_tf(cond), BG_P[cond]), batch_size=CFG["bs"], shuffle=True,
                    num_workers=CFG["workers"], drop_last=True, pin_memory=USE_AMP,
                    persistent_workers=CFG["workers"] > 0, generator=g)
    model = make_model(CFG["model"]).to(DEVICE)
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head = [p for p in model.parameters() if id(p) in head_ids]
    back = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": back, "lr": CFG["lr_backbone"]},
                             {"params": head, "lr": CFG["lr_head"]}], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"] * len(dl))
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    crit = nn.CrossEntropyLoss()
    h = {key: [] for key in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "field_loss", "field_acc"]}
    for ep in range(CFG["epochs"]):
        model.train(); tl, tc, tn = 0.0, 0, 0
        for x, y in dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); tn += len(y)
        vl, va, vp = evaluate(model, VAL_DS[k]); fl, fa, fp = evaluate(model, field_test_ds)
        for key, v in zip(h.keys(), [ep + 1, tl / tn, tc / tn, vl, va, fl, fa]): h[key].append(v)
        print(f"   epoch {ep+1}/{CFG['epochs']} | train acc {tc/tn:.3f} | lab-val acc {va:.3f} | field acc {fa:.3f}"
              f" | train loss {tl/tn:.3f}")
    json.dump(h, open(OUT / f"history/{cond}_fold{k}.json", "w"))
    np.save(OUT / f"preds/{cond}_fold{k}_val.npy", vp); np.save(OUT / f"preds/{cond}_fold{k}_field.npy", fp)
    if k == 0: torch.save(model.state_dict(), OUT / f"ckpt/{cond}_fold0.pt")
    res = dict(cond=cond, fold=k, pv_val_acc=va, pv_val_f1=f1_score(VAL_DS[k].y, vp, average="macro"),
               field_acc=fa, field_f1=f1_score(field_test_ds.y, fp, average="macro"),
               minutes=round((time.time() - t0) / 60, 2))
    del model, opt; torch.cuda.empty_cache()
    return res

# %% Step 9: Run the cross-validation (resumable)
RES_FILE = OUT / "cv_results.json"
results = json.load(open(RES_FILE)) if RES_FILE.exists() else []
done = {(r["cond"], r["fold"]) for r in results}
total = len(CFG["conds"]) * CFG["folds"]; n = 0
for c in CFG["conds"]:
    for k in range(CFG["folds"]):
        n += 1
        if (c, k) in done: continue
        print(f"[{n}/{total}] {CFG['model']} | {c} | fold {k+1}/{CFG['folds']}")
        r = run_fold(c, k); results.append(r)
        json.dump(results, open(RES_FILE, "w"), indent=1)
        print(f"   -> lab-val {r['pv_val_acc']:.3f} | FIELD {r['field_acc']:.3f} | {r['minutes']} min")
print(f"elapsed so far: {(time.time()-T0)/60:.1f} min")

# %% Step 10: Cross-validation tables (mean +/- SD) and paired comparison
df = pd.DataFrame(json.load(open(RES_FILE)))
K = CFG["folds"]; base_c, alt_c = CFG["conds"][0], CFG["conds"][-1]
ms = lambda s: f"{s.mean()*100:.2f} ± {s.std(ddof=1)*100:.2f}" if len(s) > 1 else f"{s.mean()*100:.2f}"
summ = []
for c in CFG["conds"]:
    d = df[df.cond == c].sort_values("fold")
    t = d[["fold", "pv_val_acc", "pv_val_f1", "field_acc", "field_f1"]].copy()
    t["fold"] = t.fold + 1
    for col in t.columns[1:]: t[col] = (t[col] * 100).round(2)
    t.columns = ["Fold", "Lab-val Acc (%)", "Lab-val F1 (%)", "Field Acc (%)", "Field F1 (%)"]
    print(f"\n{K}-fold cross-validation | {CFG['model']} | {c}"); print(t.to_string(index=False))
    summ.append({"Condition": c, "Lab-val Acc": ms(d.pv_val_acc), "Lab-val macro-F1": ms(d.pv_val_f1),
                 "Field Acc": ms(d.field_acc), "Field macro-F1": ms(d.field_f1),
                 "Gap (pp)": f"{(d.pv_val_acc - d.field_acc).mean()*100:.2f}"})
df.to_csv(OUT / "table_cv_folds.csv", index=False)
S = pd.DataFrame(summ); S.to_csv(OUT / "table_cv_summary.csv", index=False)
print("\nSummary (mean ± SD over folds):"); print(S.to_string(index=False))

def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj

yf = field_test_ds.y; rows = []
for k in range(K):
    p0 = np.load(OUT / f"preds/{base_c}_fold{k}_field.npy"); p1 = np.load(OUT / f"preds/{alt_c}_fold{k}_field.npy")
    ok0, ok1 = p0 == yf, p1 == yf
    b, cc = int((ok0 & ~ok1).sum()), int((~ok0 & ok1).sum())
    rows.append(dict(fold=k + 1, diff_pp=round((ok1.mean() - ok0.mean()) * 100, 2),
                     p=1.0 if b + cc == 0 else binomtest(b, b + cc, .5).pvalue))
Q = pd.DataFrame(rows); Q["p_holm"] = holm(Q.p)
Q.round(4).to_csv(OUT / "table_paired_tests.csv", index=False)
print(f"\nField accuracy, {alt_c} vs {base_c}, per fold (McNemar, Holm-corrected over {K} folds):")
print(Q.round(4).to_string(index=False))
if K >= 3:
    a = df[df.cond == base_c].sort_values("fold").field_acc.values
    b_ = df[df.cond == alt_c].sort_values("fold").field_acc.values
    print(f"Paired t-test over folds: mean gain {100*(b_-a).mean():.2f} pp, p = {ttest_rel(b_, a).pvalue:.4f}")

# %% Step 11: Confusion matrices (lab out-of-fold and field)
def cm_norm(y, p): return confusion_matrix(y, p, labels=range(NUM_CLASSES), normalize="true")

def plot_cm(mats, titles, path):
    fig, ax = plt.subplots(1, len(mats), figsize=(9 * len(mats), 8.5)); ax = np.atleast_1d(ax)
    for a, M, t in zip(ax, mats, titles):
        im = a.imshow(M, cmap="Blues", vmin=0, vmax=1)
        a.set_xticks(range(NUM_CLASSES)); a.set_yticks(range(NUM_CLASSES))
        a.set_xticklabels(short, rotation=90, fontsize=6); a.set_yticklabels(short, fontsize=6)
        a.set_title(t); a.set_xlabel("Predicted label"); a.set_ylabel("True label")
    fig.colorbar(im, ax=ax, fraction=0.02)
    plt.savefig(path, dpi=250, bbox_inches="tight"); plt.show()

y_oof = np.concatenate([VAL_DS[k].y for k in range(K)])
mats, titles = [], []
for c in CFG["conds"]:
    p_oof = np.concatenate([np.load(OUT / f"preds/{c}_fold{k}_val.npy") for k in range(K)])
    mats.append(cm_norm(y_oof, p_oof)); titles.append(f"Lab (PlantVillage, out-of-fold) | {c} | acc {accuracy_score(y_oof, p_oof)*100:.2f}%")
plot_cm(mats, titles, OUT / "figs/confusion_lab_oof.png")

mats, titles = [], []
for c in CFG["conds"]:
    M = np.mean([cm_norm(yf, np.load(OUT / f"preds/{c}_fold{k}_field.npy")) for k in range(K)], axis=0)
    mats.append(M); titles.append(f"Field (PlantDoc, mean of {K} fold models) | {c} | acc {df[df.cond==c].field_acc.mean()*100:.2f}%")
plot_cm(mats, titles, OUT / "figs/confusion_field.png")

# %% Step 12: Accuracy and loss curves (mean ± SD over folds)
E = CFG["epochs"]; xs = np.arange(1, E + 1)
H = {c: [json.load(open(OUT / f"history/{c}_fold{k}.json")) for k in range(K)] for c in CFG["conds"]}
arr = lambda c, key: np.array([h[key] for h in H[c]])

def band(ax, a, label):
    m = a.mean(0); s = a.std(0, ddof=1) if a.shape[0] > 1 else np.zeros_like(m)
    ax.plot(xs, m, marker="o", ms=3, label=label); ax.fill_between(xs, m - s, m + s, alpha=0.2)

for c in CFG["conds"]:
    fig, ax = plt.subplots(2, 1, figsize=(7, 8.5))
    band(ax[0], arr(c, "train_acc"), "Train accuracy (augmented)")
    band(ax[0], arr(c, "val_acc"), "Validation accuracy (PlantVillage, lab)")
    band(ax[0], arr(c, "field_acc"), "Test accuracy (PlantDoc, field)")
    ax[0].set_title(f"Model Accuracy ({CFG['model']}, {c})"); ax[0].set_ylim(0, 1.02)
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy"); ax[0].grid(True); ax[0].legend(fontsize=8, loc="center right")
    band(ax[1], arr(c, "train_loss"), "Train loss")
    band(ax[1], arr(c, "val_loss"), "Validation loss (PlantVillage)")
    band(ax[1], arr(c, "field_loss"), "Test loss (PlantDoc)")
    ax[1].set_title(f"Model Loss ({CFG['model']}, {c})")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss"); ax[1].grid(True); ax[1].legend(fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig(OUT / f"figs/curves_{c}.png", dpi=300); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for c in CFG["conds"]:
    band(ax[0], arr(c, "val_acc"), c); band(ax[1], arr(c, "field_acc"), c)
ax[0].set_title("Lab validation accuracy (PlantVillage)"); ax[1].set_title("Field test accuracy (PlantDoc)")
for a in ax: a.set_xlabel("Epoch"); a.set_ylabel("Accuracy"); a.grid(True); a.legend()
plt.tight_layout(); plt.savefig(OUT / "figs/curves_compare.png", dpi=300); plt.show()

# %% Step 13: Interpretability with Grad-CAM (fold-0 models) and foliage attention ratio
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def load_model(cond):
    m = make_model(CFG["model"], NUM_CLASSES, pretrained=False)
    m.load_state_dict(torch.load(OUT / f"ckpt/{cond}_fold0.pt", map_location="cpu"))
    return m.to(DEVICE).eval()

def target_layers(m): return [m.layer4[-1]] if CFG["model"].startswith("resnet") else [m.blocks[-1]]

def grabcut_mask(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); m = np.zeros(bgr.shape[:2], np.uint8); h, w = m.shape
    try:
        cv2.grabCut(bgr, m, (int(.1*w), int(.1*h), int(.8*w), int(.8*h)),
                    np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None
    mk = ((m == 1) | (m == 3)).astype(np.float32)
    return mk if 0.05 < mk.mean() < 0.95 else None

mods = {c: load_model(c) for c in [base_c, alt_c]}
cams = {c: GradCAM(model=mods[c], target_layers=target_layers(mods[c])) for c in mods}
idx = np.random.RandomState(0).choice(len(field_test_ds), min(80, len(field_test_ds)), replace=False)
ratio = {c: [] for c in cams}; shown = []
for i in idx:
    rgb = np.array(field_test_ds.imgs[i]); mk = grabcut_mask(rgb)
    if mk is None: continue
    xt = EVAL_TF(field_test_ds.imgs[i]).unsqueeze(0).to(DEVICE)
    maps, preds = {}, {}
    for c, cam in cams.items():
        with torch.no_grad(): preds[c] = int(mods[c](xt).argmax(1))
        hm = cam(input_tensor=xt, targets=None)[0]
        maps[c] = hm; ratio[c].append(float((hm * mk).sum() / (hm.sum() + 1e-8)))
    if len(shown) < 6: shown.append((rgb, maps, preds, int(field_test_ds.y[i])))

print(f"images with a valid GrabCut mask: {len(ratio[base_c])}")
for c in ratio: print(f"mean foliage-attention ratio {c}: {np.mean(ratio[c]):.3f} ± {np.std(ratio[c]):.3f}")
if len(ratio[base_c]) > 5:
    print(f"Wilcoxon p ({base_c} vs {alt_c}):", wilcoxon(ratio[base_c], ratio[alt_c]).pvalue)
print("NOTE: GrabCut mask is an approximation; Grad-CAM explains the PREDICTED class (shown in the titles).")

fig, ax = plt.subplots(3, len(shown), figsize=(3.2 * len(shown), 10))
for j, (rgb, maps, preds, true) in enumerate(shown):
    ax[0, j].imshow(rgb); ax[0, j].set_title(f"true: {short[true]}", fontsize=6)
    for r, c in zip([1, 2], [base_c, alt_c]):
        ax[r, j].imshow(show_cam_on_image(rgb / 255.0, maps[c], use_rgb=True))
        ax[r, j].set_title(f"{c}: {short[preds[c]]} ({'correct' if preds[c] == true else 'wrong'})", fontsize=6)
    for r in range(3): ax[r, j].axis("off")
plt.tight_layout(); plt.savefig(OUT / "figs/gradcam.png", dpi=250); plt.show()

# %% Step 14: Package the results
zip_path = Path("/kaggle/working") / f"results_{OUT.name}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file() and "ckpt" not in p.parts:
            z.write(p, p.relative_to(OUT.parent))
print("Saved:", zip_path, f"({zip_path.stat().st_size/1e6:.1f} MB)")
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.suffix in {".csv", ".png"}: print(p)
print(f"TOTAL TIME: {(time.time()-T0)/60:.1f} min")
