# Lab-to-Field Gap in Plant Disease Classification (PlantVillage → PlantDoc)

Code for the paper **"Quantifying and Partially Mitigating the Lab-to-Field Gap in Plant Disease Classification with Lightweight CNNs"**
(Author : Fahad Bin Aref)

## What this repository does
Three lightweight CNNs (MobileNetV3-Large, EfficientNet-B0, ResNet18) are fine-tuned on **PlantVillage** and tested on
**PlantDoc** over 27 shared crop–disease classes. Seven training conditions are compared (C0–C5 and an oracle, UB), with
three seeds, McNemar tests with Holm correction, five-fold cross-validation, training curves, confusion matrices and Grad-CAM.

## Headline results (see the paper for all tables)
| Setting (EfficientNet-B0) | PlantVillage val. acc. | PlantDoc field acc. |
|---|---|---|
| C0 baseline (3 seeds) | 98.77 ± 0.09 | 16.21 ± 0.98 |
| C3 background + photometric + erasing (3 seeds) | 98.49 ± 0.37 | 24.71 ± 1.88 |
| UB oracle with labelled field images (3 seeds) | 98.82 ± 0.08 | 58.89 ± 0.33 |
| C0, 5-fold CV | 98.22 ± 0.20 | 16.07 ± 1.47 |
| C3, 5-fold CV | 97.57 ± 0.39 | 22.48 ± 1.50 |

## Repository structure
```
notebooks/01_full_study.ipynb              3 models x 6 conditions + oracle x 3 seeds, statistics, efficiency, Grad-CAM
notebooks/02_fast_cv_curves_gradcam.ipynb  EfficientNet-B0, C0 vs C3: 5-fold CV, curves, confusion matrices, Grad-CAM
notebooks/extras/step19_training_curves_cell.py   optional extra cell for per-epoch curves in notebook 01
scripts/                                   the same code as plain .py files, plus make_paper_figures.py (Fig. 1 and 4)
results/   figures/   docs/                 output tables, figures, paper
```

## Data (not included)
Download from Kaggle and attach to the notebook (check each dataset's licence before reuse):
- PlantVillage: `abdallahalidev/plantvillage-dataset` (color and segmented folders)
- PlantDoc: `nirmalsankalana/plantdoc-dataset`

## How to reproduce (Kaggle)
1. Create a Kaggle notebook, import `notebooks/01_full_study.ipynb`, set **Accelerator = GPU** and **Internet = ON**.
2. Add the two datasets above as input data.
3. Run with `QUICK = True` first (smoke test), then set `QUICK = False` and run all cells (estimated 3–5 hours of GPU for the main study).
4. Repeat with `notebooks/02_fast_cv_curves_gradcam.ipynb` (`SMOKE = False`, about 30–60 minutes).
5. Tables and figures are written to `/kaggle/working/full` and `/kaggle/working/fast`.
Seeds: 42, 123, 2024 (main study); 42 + fold index (cross-validation). Settings are listed in Table III of the paper.

## Which code produces which paper item
| Paper item | Notebook and step |
|---|---|
| Table IV, Fig. 4 | 01 · Step 12 (Table A); Fig. 4: `scripts/make_paper_figures.py` |
| Table V, Section IV-B | 01 · Step 13 |
| Table VI (oracle share) | computed from Table IV (UB rows) |
| Table VIII | 01 · Step 14 |
| Section IV-G (foliage attention ratio) | 01 · Step 16 |
| Tables VII, Figs. 5–6, 7–8, 9 | 02 · Steps 10, 12, 11, 13 |

## Citation
```bibtex
@article{[key],
  title   = {Quantifying and Partially Mitigating the Lab-to-Field Gap in Plant Disease Classification with Lightweight CNNs},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]}
}
```
See also `CITATION.cff`. Licence: MIT (code only; datasets keep their own licences).
