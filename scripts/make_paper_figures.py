"""Re-creates Fig. 1 (workflow) and Fig. 4 (field accuracy by condition) of the paper.
Fig. 4 uses the Table IV values (mean and SD over three seeds). Run: python scripts/make_paper_figures.py
MobileNetV3-Large C1 is left out until its re-run is finished (set the value and SD below, then re-run)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path("figures"); OUT.mkdir(exist_ok=True)

# ---------------- Fig. 1: workflow ----------------
fig, ax = plt.subplots(figsize=(10, 3.7)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, text, fc="#E8F0FA", ec="#2F5597", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)
def arrow(x1, y1, x2, y2, style="-", color="#444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color, linestyle=style))
box(1, 24, 16, 12, "PlantVillage\n(lab images)\n54,305 images\n38 classes", fc="#E2F0D9", ec="#548235")
box(1, 4, 16, 12, "PlantDoc\n(web/field images)\n2,922 images\n28 folders", fc="#FCE4D6", ec="#C55A11")
box(22, 14, 14, 12, "Class mapping\n27 shared classes\n(13 crops)", fc="#FFF2CC", ec="#BF9000")
box(41, 24, 17, 12, "Training pool\nstratified split\nor 5-fold CV")
box(41, 4, 17, 12, "Field split\nfield-val (876)\nfield-test (2,044)", fc="#FCE4D6", ec="#C55A11")
box(63, 24, 14, 12, "Augmentation\nC0 to C5, UB\n(Table II)")
box(82, 24, 17, 12, "Fine-tune CNNs\nMobileNetV3-L,\nEfficientNet-B0,\nResNet18", fs=7.4)
box(82, 4, 17, 12, "Evaluation\nlab-val, field-test,\nMcNemar, CV,\nGrad-CAM", fc="#EADCF4", ec="#7030A0", fs=7.6)
arrow(17.5, 28, 22.5, 22); arrow(17.5, 12, 22.5, 18); arrow(36.5, 22, 41, 28); arrow(36.5, 18, 41, 12)
arrow(58.5, 30, 63, 30); arrow(77.5, 30, 82, 30); arrow(90.5, 23.4, 90.5, 16.6); arrow(58.5, 8, 82, 8)
ax.text(70, 9.2, "field-test (final reporting only)", fontsize=7, ha="center", color="#444")
arrow(49.5, 16.6, 49.5, 23.4, style="--", color="#C55A11")
ax.text(50.8, 20, "field-val\n(UB only)", fontsize=6.6, color="#C55A11", va="center")
plt.savefig(OUT / "fig1_workflow.png", dpi=300, bbox_inches="tight"); plt.close()

# ---------------- Fig. 4: field accuracy by condition (Table IV) ----------------
conds = ["C0", "C1", "C2", "C3", "C4", "C5", "UB"]
field = {"MobileNetV3-L": [18.17, None, 22.03, 22.78, 16.86, 22.06, 58.51],
         "EfficientNet-B0": [16.21, 18.87, 19.31, 24.71, 16.44, 18.35, 58.89],
         "ResNet18": [23.47, 23.43, 25.28, 24.93, 23.48, 25.93, 60.09]}
sd = {"MobileNetV3-L": [1.36, None, 1.44, 0.35, 0.86, 0.32, 1.14],
      "EfficientNet-B0": [0.98, 1.08, 0.50, 1.88, 0.87, 1.10, 0.33],
      "ResNet18": [1.72, 1.11, 1.26, 0.67, 0.87, 0.93, 1.21]}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5})
fig, ax = plt.subplots(figsize=(3.45, 2.7))
w = 0.27; x = np.arange(len(conds)); cols = ["#4472C4", "#ED7D31", "#70AD47"]
for i, (m, v) in enumerate(field.items()):
    for j, val in enumerate(v):
        if val is None: continue
        ax.bar(x[j] + (i - 1) * w, val, w, yerr=sd[m][j], color=cols[i], capsize=1.2, edgecolor="white",
               linewidth=0.3, label=m if j == 0 else None, error_kw=dict(lw=0.6))
ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylabel("Field-test accuracy (%)"); ax.set_xlabel("Training condition")
ax.set_ylim(0, 72); ax.grid(axis="y", alpha=0.3, lw=0.4)
ax.legend(frameon=False, fontsize=5.8, ncol=3, loc="upper left", handlelength=1.0, columnspacing=0.8)
plt.tight_layout(pad=0.4); plt.savefig(OUT / "fig4_field_accuracy.png", dpi=400); plt.close()
print("saved figures to", OUT.resolve())
