"""Produce a self-contained notebook and archive for the completed image study."""
from pathlib import Path
import ast
import json
import os
import zipfile
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parent
source = (ROOT/'image_pipeline.py').read_text(encoding='utf-8')
source = source[:source.index("if __name__=='__main__':")]
source = source.replace('ROOT = Path(__file__).resolve().parent','ROOT = Path.cwd()')
cells = []
def md(s): cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):
    ast.parse(s)
    cells.append(nbformat.v4.new_code_cell(s))

md('''# AquaSense: 500 synthetic strip images — complete workflow

**Proposed design only. All images, concentrations and performance results are synthetic.**

This notebook contains all code for image rendering, preprocessing, feature extraction, model training, evaluation and inference. No project `.py` imports are needed. It demonstrates desktop Python processing; it does not deploy models to an ESP32.

Extract the whole bundle, open this notebook in its folder, and run cells in order. The default uses the supplied images and fitted models. Change `REBUILD = True` below to regenerate and retrain everything. The full run can take several minutes or longer.

The dataset has 500 JPEGs (640 × 480), from 250 simulated sample profiles with two captures each. Five profiles share a session; complete sessions stay together in train or test. The split contains 400 training images (200 profiles, 40 sessions) and 100 test images (50 profiles, 10 sessions).

`datasets/images/` contains raw JPEGs, `datasets/preprocessed/` contains corrected images, and the CSVs in `datasets/` hold labels, extracted features and QC. `models/`, `results/` and `charts/` contain the corresponding outputs.
''')
md('## 1. Optional package installation\nSet the switch only when dependencies are missing. The pinned versions match this run. Restart the kernel if installation changes imported packages.')
requirements = (ROOT/'requirements.txt').read_text().splitlines()
code('INSTALL_DEPENDENCIES = False\nif INSTALL_DEPENDENCIES:\n    import subprocess, sys\n    subprocess.check_call([sys.executable,"-m","pip","install",*'+repr(requirements)+'])')
md('## 2. Imports, fixed seed and proposed geometry\nSeven reagent spots are assumed to respond primarily to their corresponding metals. These response functions and concentration ranges are invented, not laboratory-calibrated. HSV/LAB are derived representations of RGB.')
code(source[:source.index('def prepare_folders')])
code('from IPython.display import display, Image, Markdown\nprint("Working folder:", ROOT)')
sections = [
    ('3. Synthetic scene generation','prepare_folders','preprocess_image',
     'The generative concept photo is a visual reference. Dataset scenes are rendered independently from explicit geometry and equations, allowing traceable concentration labels. Neither labels nor concentrations are printed on images.'),
    ('4. Image preprocessing and feature extraction','preprocess_image','model_spaces',
     'The preprocessor reads image pixels only. It detects fiducials, rectifies perspective, denoises, uses black/white patches for affine color correction, checks gray-reference residuals, and extracts nine color values for each of seven spots. Rendering metadata is excluded.'),
    ('5. Four model families, grouped tuning and prediction','model_spaces','make_charts',
     'Feature and target scalers are fitted inside training folds. All families use identical five-fold grouped training validation and the same held-out sessions. Predictions are not clipped for either validation or test scoring. This is a different measurement design from the earlier single-RGB simulation.'),
    ('6. Charts, audits and reproducibility','make_charts',None,
     'Audits check image counts, dimensions, feature extraction, group isolation during training and the saved metrics. The manifest records file hashes and package versions.')]
for title,start,end,description in sections:
    md(f'## {title}\n{description}')
    block = source[source.index('def '+start):source.index('def '+end) if end else len(source)]
    code(block)
md('## 7. Run or load the study\nThe saved models are already trained. Keep `REBUILD = False` for the ready-to-use workflow. Regeneration overwrites outputs in this extracted bundle.')
code('''REBUILD = False
required = [ROOT/'datasets/labels.csv', ROOT/'datasets/features.csv', ROOT/'results/test_metrics.csv']
required += [ROOT/'models'/f'{metal}_{model}.joblib' for metal in METALS for model in ['PLS','RandomForest','XGBoost','RBF_SVR']]
if REBUILD or not all(path.exists() for path in required):
    run_all()
else:
    print('Loading supplied synthetic images and trained models.')
labels = pd.read_csv(ROOT/'datasets/labels.csv')
features = pd.read_csv(ROOT/'datasets/features.csv')
metrics = pd.read_csv(ROOT/'results/test_metrics.csv')
verify_study(labels,features)
display(labels.head())
display(features.head())
display(pd.read_csv(ROOT/'results/model_summary.csv'))
display(metrics)
''')
md('## 8. Proposed design, actual dataset samples and preprocessing\nThe concept image is not a prototype photograph. Numerical scenes use the same seven-spot layout, but do not use its pixels as training data.')
code('''for path in ['design/proposed_strip_concept.png','design/numerical_template.png','charts/image_contact_sheet.png','charts/preprocessing_steps.png','charts/model_comparison.png']:
    display(Image(filename=str(ROOT/path)))
''')
md('## 9. From one JPEG to concentration estimates\nThis example uses a held-out synthetic image. The path is the only input to preprocessing and prediction. Real images are not supported without calibrating and validating the actual strip and camera pipeline.')
code('''example = labels[labels.partition=='test'].iloc[0]
predicted, qc = predict_image(ROOT/example.image_path)
display(predicted)
print('Image QC:', qc)
''')
md('''## 10. Research interpretation and hardware pathway

The simulated pad selectivity is built into the generator. Good predictive scores would indicate recovery of that assumed relationship, not proof of chemical selectivity, detection limits or field accuracy. This dataset has 250 concentration profiles, not 500 independent water samples. All images share the same template and simulator; grouped testing cannot measure the gap to real images.

The current route is ESP32 capture -> transfer image to a processing computer/server -> preprocess -> model predicts concentrations -> return results. An ESP32-S3 firmware draft is supplied separately, but its board-specific camera pins are not configured or hardware-tested. No on-device inference conversion or hardware benchmark is supplied.

For the real project, finalize the reagent design and geometry, collect images paired with reference-laboratory concentrations across lots, sessions and interferents, and reserve an external test campaign. Replace the simulator and assumed coefficients with measured data. Keep repeated images of one sample together in splitting. Validate the entire camera-to-result process.
''')
md('## 11. Interpret the results\nThe table below ranks the four families on the common held-out **synthetic** test sessions. Per-metal MAE and RMSE remain in simulated µg/L; R² is not classification accuracy. Test ranking is descriptive and was not used to choose the newer prototype models.')
code('''from IPython.display import Markdown
interpretation_text = (ROOT/'results/result_interpretation.md').read_text(encoding='utf-8')
display(Markdown(interpretation_text))
''')
notebook = nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13.1'}})
nbformat.validate(notebook)
nbformat.write(notebook,ROOT/'AquaSense_500_Images.ipynb')

# Execute code cells sequentially in one fresh namespace, including pixel-to-model inference.
os.chdir(ROOT)
namespace = {'__name__':'__notebook__'}
for i,cell in enumerate(cells):
    if cell.cell_type=='code':
        exec(compile(cell.source,f'notebook_cell_{i}','exec'),namespace)
assert len(namespace['predicted'])==28

summary = pd.read_csv(ROOT/'results/model_summary.csv')
report = '# Synthetic image study results\n\nAll figures describe simulated images and assumed chemistry.\n\n'
report += '500 images; 250 profiles; 50 sessions. Train: 400 images, 200 profiles, 40 sessions. Test: 100 images, 50 profiles, 10 sessions.\n\n'
report += '| Model | Mean normalized RMSE (% full scale) | Mean R² |\n|---|---:|---:|\n'
for row in summary.itertuples():
    report += f'| {row.model} | {row.mean_NRMSE_percent_full_scale:.3f} | {row.mean_R2:.3f} |\n'
report += '\nThese are post-evaluation comparisons, not validated deployment selections. The single-split experiment does not establish statistical superiority. Reagent-specific responses are assumed, and none of these results support real-water detection claims. See README.md for full limitations and differences from previous experiments.\n'
(ROOT/'results/study_report.md').write_text(report,encoding='utf-8')
namespace['write_manifest']()
archive_path = ROOT.parent/'AquaSense_500_Synthetic_Images.zip'
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.py', '.pyc']:
            archive.write(path,path.relative_to(ROOT.parent))
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert sum('/datasets/images/' in name and name.endswith('.jpg') for name in archive.namelist())==500
    assert sum(name.endswith('.ipynb') for name in archive.namelist())==1
print(f'Notebook cells validated and run; ZIP verified: {archive_path} ({archive_path.stat().st_size/1048576:.2f} MB)')
