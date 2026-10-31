# AquaSense — simulated image-to-concentration ML prototype

**Research preparation only.** Every strip image and concentration in this repository is simulated. No physical strip, ESP32-S3 capture, or laboratory reference measurement has been used. The reported scores measure how well models recover assumptions built into the simulator; they are **not real-world heavy-metal detection accuracy**.

## Run the complete workflow

Open [AquaSense_All_In_One.ipynb](image_study_500/AquaSense_All_In_One.ipynb) in Jupyter or Google Colab and run all cells. The notebook is self-contained: it generates synthetic images, preprocesses them, extracts RGB/HSV/LAB features, trains and compares PLS Regression, Random Forest, XGBoost, and RBF SVR for Pb, Hg, Cd, As, Cr, Cu, and Zn, then saves figures and CSV results. Its first cell can install missing packages. A fresh run needs substantial time and disk space; later runs reuse generated outputs. Set `REBUILD_ALL = True` in the first code cell to retrain from scratch.

For a local Python environment:

```powershell
cd image_study_500
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt notebook
.venv\Scripts\python -m jupyter notebook AquaSense_All_In_One.ipynb
```

To run the optional desktop image-upload app **after** the notebook has generated the v2 models:

```powershell
.venv\Scripts\python app.py
```

See the [detailed project guide](image_study_500/README.md) for the data-generation assumptions, grouped splits, image QC, ESP32 integration boundary, and real-data collection plan.

## Repository contents

| Path | Purpose |
| --- | --- |
| `image_study_500/AquaSense_All_In_One.ipynb` | Main reproducible workflow and model training |
| `image_study_500/*.py` | Modular implementation, packaging, app, and tests |
| `image_study_500/datasets/` | Baseline synthetic labels and extracted features |
| `image_study_500/scenario_study/` | Expanded scenario labels, results, and documentation |
| `image_study_500/improved_study/` | v2 audit results, comparison charts, and [report](image_study_500/improved_study/REPORT.md) |
| `image_study_500/real_data/` | Templates and collection plan for future measured data |
| `image_study_500/firmware/` | Draft ESP32-S3 image-capture integration; board pin mapping remains to be configured |

Generated JPEGs, corrected images, model binaries, local captures, and portable ZIP archives are omitted from Git to keep the repository practical to clone. The notebook regenerates training data and models. The published CSVs and charts are **synthetic demonstration results**.

## Evaluation boundary

The simulator uses an assumed seven-spot color response, shared card layout, lighting and camera variations, and session grouping. v2 selects models using grouped training-only validation and audits them on a newly generated synthetic set. The [v2 report](image_study_500/improved_study/REPORT.md) shows only modest improvements; paired uncertainty intervals span zero. Its 90% prediction intervals lose coverage under simulated shift, and the domain gate does not reliably catch that failure.

Before any real detection claim, collect images of manufactured strips under the intended camera/lighting setup, pair each sample with reference-laboratory concentrations, validate on independent manufacturing batches and acquisition days, and recalibrate or retrain. See [real-data collection plan](image_study_500/real_data/COLLECTION_PLAN.md).
