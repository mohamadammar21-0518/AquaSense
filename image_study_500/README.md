# AquaSense: all-in-one synthetic image research prototype

**Proposed design and simulation only. No physical strip, real camera capture or experimentally measured concentration is represented.**

This study demonstrates the workflow: image -> detect alignment marks -> perspective correction -> reference-patch color correction -> extract RGB/HSV/LAB from seven spots -> regress seven concentrations. A generative concept image is supplied in `design/`; the training images are independently rendered mathematical scenes based on the same layout, not recolored copies of that photograph. Every target value is chosen before image rendering.

## Files and quick start

The current prediction bundle is **expanded-v2** in `improved_study/models/`. It selects a model and color-feature recipe for each metal using grouped training validation, with 24 separate sessions for synthetic interval calibration. The all-in-one notebook builds this bundle automatically when it is missing. The upload app and notebook sample prediction both use v2. Read `improved_study/REPORT.md` for comparisons, fresh-audit results, interval coverage and remaining limitations. Earlier study results remain available for comparison.

For the simplest workflow, use **`AquaSense_All_In_One.ipynb` by itself**. Put the notebook in a writable empty folder, open it in Jupyter or Colab, and Run All. It detects the absence of saved outputs, generates both synthetic datasets, trains the models, evaluates them and writes the results folders automatically. If required packages are missing, its first cell installs them. A complete fresh run takes considerable time and disk space. Later runs load those outputs; set **`REBUILD_ALL = True`** in the first code cell to force a fresh training run. The ZIP additionally supplies already-trained outputs for quicker inspection but is not required for the notebook to run. The older separate notebooks are retained in the workspace for provenance.

- `datasets/images/`: exactly 500 JPEG images, 640 x 480 pixels.
- `datasets/labels.csv`: image paths, sample/session IDs, simulated concentrations and split labels.
- `datasets/features.csv`: 63 image-extracted color features plus IDs and split. No concentration labels are used in feature extraction.
- `datasets/preprocessed/`: 500 rectified, reference-corrected card images.
- `datasets/preprocessing_qc.csv`: marker detection and reference-patch residuals.
- `datasets/render_metadata.json`: rendering parameters for audit only; the preprocessor does not read them.
- `models/`: 28 fitted pipelines, four families for each of seven metals.
- `results/`: test metrics, predictions, validation search results and methodological report.
- `results/result_interpretation.md`: paper-ready explanation of the metrics, model ranking, and image-to-result decision flow.
- `AquaSense_Scenario_Study.ipynb`, `scenario_study/`: a separate 1,800-image synthetic robustness experiment with scenario-wise model errors and image-rejection checks. See `scenario_study/README.md`. Its results must not be mixed with the original 500-image test figures.
- `improved_model.py`, `improved_study/`: revised training, all four model families, a new 200-image audit, comparison tables and v2 prediction models. Run `python improved_model.py` to rebuild only this stage from existing scenario data.
- `app.py`, `interpretation.py`, `quality_control.py`, `reliability.py`: the optional local upload prototype with per-image interpretation and conservative withholding. Run `python app.py` after installing `requirements.txt`.
- `charts/`: example images, preprocessing steps, and model comparisons.
- `design/proposed_strip_concept.png`: AI-generated visual concept only.
- `design/numerical_template.png`: exact canonical layout used by the renderer.

## Dataset independence and split

There are **250 independently generated concentration profiles**, with **two camera captures of each**, for 500 images. Five profiles share one simulated acquisition session, making 50 sessions of ten images. The split keeps complete sessions together: 40 sessions / 200 profiles / 400 images for training, and 10 sessions / 50 profiles / 100 images for testing. All four models receive exactly the same partition and five grouped training-validation folds. The 500 images must not be described as 500 independent water samples.

## Proposed card design

A white imaging card holds seven circular reaction spots, in the fixed order Pb, Hg, Cd, As, Cr, Cu, Zn. Four black square fiducials support alignment. White, gray and black patches support a simple color correction and a check of residual error. The concept is a starting point for your team, not a validated strip blueprint. The renderer uses 560 x 280 canonical pixels; it does not specify physical manufacturing dimensions.

The seven pads have assumed dominant responses to their respective metals, small cross-responses, saturation, simulated pH and temperature effects and batch variation. None of these coefficients are measured chemistry. Selectivity is built into the simulator; high scores would reflect recovery of that assumed relationship. Seven RGB spots provide 21 underlying color channels, unlike the earlier one-RGB-triplet experiment. Consequently these scores are not directly comparable with the earlier 720/1,500-row study.

## What varies in the images

Concentration-dependent pad colors, paper/pad texture, card rotation, scale and mild perspective, illumination strength, channel white balance, lighting gradient, camera-like gamma variation, sensor noise, Gaussian blur and JPEG quality. These are generic camera effects, not a calibrated ESP32-CAM noise model. The concentration values never appear as text or filenames in images.

Preprocessing detects four dark square marks directly from image pixels and estimates a homography. It uses a fixed known layout to sample the reference patches and inner spot regions. Per-channel affine correction uses black/white references, while gray residuals provide QC. HSV and LAB are deterministic transforms of the median corrected RGB in each spot; they do not add independent observations. No true rendering pose or concentration is supplied to preprocessing. The detector assumes modest rotation and a visible complete card; severe occlusion, arbitrary rotation and field scenes are outside this dataset.

## Models and limits

PLS Regression, Random Forest, XGBoost and RBF SVR use fitted feature scaling and target scaling inside pipelines. Target scaling is fitted within training folds; hyperparameters are selected by validation RMSE in original concentration units. Raw regression predictions are used consistently for validation and test metrics. This differs from the earlier clipped-output feature-only demo. Negative outputs, if any, are reported as a diagnostic rather than silently modified. Per-metal metrics and macro RMSE normalized by configured simulation maxima are provided, with a training-mean baseline. Test-set model rankings are descriptive and do not choose a validated deployment model.

Preprocessing constants and the known geometric layout are set before evaluating test predictions. Grouping blocks repeat-capture/session leakage, but train and test still share the same simulator and template. This is a software proof of concept, not an analytical validation. A single AI concept picture cannot supply known concentrations or chemistry. Real images paired with reference-laboratory measurements are required for detection limits, recovery, selectivity and field accuracy.

## ESP32 integration

The image comes first; concentrations are model outputs. The demonstrated route is ESP32 camera capture -> image transfer to a processing computer/server -> image preprocessing -> model inference -> return concentrations. This code runs on desktop Python. It does not deploy scikit-learn/XGBoost to an ESP32 or verify its memory, latency or power budget. On-device inference requires a separate hardware-compatible implementation and validation. Use a fixed holder and controlled lighting when you develop the real strip.

## Research wording

Describe this as a **simulation-based evaluation of a seven-spot image-processing and regression pipeline**, with 500 rendered images from 250 synthetic profiles in 50 acquisition groups. Do not call the concept a manufactured device or report these results as experimental detection accuracy.
