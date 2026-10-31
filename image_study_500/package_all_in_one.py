"""Create one self-contained training notebook and one complete research ZIP."""
from pathlib import Path
import ast
import re
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / 'AquaSense_All_In_One.zip'


def source_without_main(filename, imports=()):
    source = (ROOT / filename).read_text(encoding='utf-8')
    for marker in ("if __name__ == '__main__':", "if __name__=='__main__':"):
        if marker in source:
            source = source[:source.index(marker)]
            break
    for line in imports:
        assert line in source, (filename, line)
        source = source.replace(line, '')
    ast.parse(source)
    return source


base = source_without_main('image_pipeline.py', ['ROOT = Path(__file__).resolve().parent'])
quality = source_without_main('quality_control.py', [
    'from image_pipeline import ROOT, SPOTS, REF_CENTERS, preprocess_image\n'])
reliable = source_without_main('reliability.py', [
    'from image_pipeline import ROOT, SEED, METALS, MAXIMA, TARGETS, FEATURES, model_spaces\n',
    'from quality_control import check_image, calibrate_quality, ImageRejected\n'])
scenario = source_without_main('scenario_study.py', [
    'from image_pipeline import ROOT, METALS, TARGETS, MAXIMA, FEATURES, assay_colors, render_card, camera_capture, model_spaces, preprocess_image\n',
    'from quality_control import ImageRejected, check_image\n'])
# The original study and scenario generator use different seeds. Preserve the
# original module global so the first study and its model spaces reproduce.
scenario = re.sub(r'\bSEED\b', 'SCENARIO_SEED', scenario)
interpretation = source_without_main('interpretation.py', [
    'from image_pipeline import ROOT, METALS, MAXIMA\n'])
improved = source_without_main('improved_model.py', [
    'from image_pipeline import ROOT, METALS, MAXIMA, TARGETS, FEATURES, CHANNELS, model_spaces, lab_from_rgb, assay_colors, render_card, camera_capture\n',
    'from quality_control import check_image, ImageRejected\n',
    'from reliability import fit_domain\n'])


def md(value):
    return nbformat.v4.new_markdown_cell(value)


def code(value):
    ast.parse(value)
    return nbformat.v4.new_code_cell(value)


cells = [
    md('''# AquaSense — all-in-one synthetic image study

**Research preparation only. All images, color responses, concentrations, intervals, and model scores are simulated.** There is no manufactured strip or verified ESP32-S3 dataset here.

This single notebook contains the source for both studies: the original 500-image baseline and the expanded 1,800-image scenario experiment. It covers rendering, image preprocessing/QC, RGB–HSV–LAB feature extraction, four-model tuning and training, grouped testing, calibration, charts, reporting, and one-image prediction. It does not import project `.py` files or need pre-existing datasets/models.

Download this notebook alone into an empty working folder, open it in Jupyter or Colab, and **Run All**. On a fresh folder it automatically creates `datasets/`, `models/`, `results/`, `charts/` and `scenario_study/`, then trains and evaluates everything. A full run can take considerable time and disk space. Later runs load those saved outputs unless `REBUILD_ALL = True` is set. Keep the original and expanded results separate when writing a paper.'''),
    md('## 1. Controls and optional dependency setup'),
    code('''from pathlib import Path
import importlib.util
import subprocess
import sys
ROOT = Path.cwd()
expected_outputs = [ROOT/'datasets/labels.csv', ROOT/'results/test_metrics.csv',
                    ROOT/'models/reliable/config.json', ROOT/'scenario_study/labels.csv',
                    ROOT/'scenario_study/results/metrics_by_scenario_metal.csv']
REBUILD_ALL = not all(path.exists() for path in expected_outputs)
# To force retraining later, change REBUILD_ALL to True here.
packages = {'numpy':'numpy==2.2.4', 'pandas':'pandas==2.2.3', 'scipy':'scipy==1.15.3',
            'sklearn':'scikit-learn==1.6.1', 'xgboost':'xgboost==3.4.1',
            'matplotlib':'matplotlib==3.10.3', 'joblib':'joblib==1.5.0',
            'cv2':'opencv-python==5.0.0.93', 'threadpoolctl':'threadpoolctl==3.6.0'}
missing_packages = [spec for module, spec in packages.items() if importlib.util.find_spec(module) is None]
if missing_packages:
    print('Installing missing packages:', missing_packages)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
print('Project folder:', ROOT)
print('Rebuild all:', REBUILD_ALL)
'''),
    md('## 2. Image simulator, preprocessing and four regression families\nThe seven response functions are invented assumptions. Two captures of one profile are not two independent water samples. HSV and LAB are deterministic transforms of measured RGB values.'),
    code(base),
    md('## 3. Image-quality gate and synthetic-only prediction intervals\nQC rejects unreadable or unsupported captures. The later prediction intervals are calibrated on independent synthetic sessions and must not be called field confidence intervals, detection limits, or safety thresholds.'),
    code(quality),
    code(reliable),
    md('## 4. Expanded scenarios\nNew profiles cover standard, low-range and mixed-extreme concentrations. An entirely unseen synthetic camera/batch shift is reserved as a challenge. Deliberately damaged images audit rejection, not regression training.'),
    code(scenario),
    md('## Improved models: feature selection and robustness\nVersion 2 selects each metal’s model and color-feature recipe using grouped training validation. It reserves 24 independent development sessions for synthetic interval calibration. A fresh 200-image audit is generated only after selection. Existing held-out results are reported as exploratory because they have already been inspected.'),
    code(improved),
    md('## 5. Plain-language interpretation'),
    code(interpretation),
    md('## 6. Run or load everything\nA fresh folder triggers the full experiment automatically. To force retraining later, set `REBUILD_ALL = True` in section 1. This first trains the baseline and its QC/interval bundle, then independently generates and trains the expanded scenario study. Test/challenge data never tune models.'),
    code('''if REBUILD_ALL:
    run_all()                     # Original 500-image baseline
    load_reliable.cache_clear()
    train_reliable()              # Training-only model selection and synthetic interval calibration
    main()                        # Expanded 1,800-image scenario study + 120 QC stress cases
else:
    needed = [ROOT/'datasets/labels.csv', ROOT/'results/test_metrics.csv',
              ROOT/'models/reliable/config.json', ROOT/'scenario_study/labels.csv',
              ROOT/'scenario_study/results/metrics_by_scenario_metal.csv']
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError('Saved results are incomplete. Set REBUILD_ALL=True. Missing: ' + ', '.join(missing))
    print('Loaded the supplied images, trained models and saved results.')

improved_outputs = [V2_MODELS/'config.json', V2_DIR/'comparison_metrics.csv', V2_DIR/'REPORT.md', V2_DIR/'comparison.png']
if REBUILD_ALL or not all(p.exists() for p in improved_outputs):
    run_improvement()

baseline_labels = pd.read_csv(ROOT/'datasets/labels.csv')
expanded_labels = pd.read_csv(ROOT/'scenario_study/labels.csv')
baseline_metrics = pd.read_csv(ROOT/'results/test_metrics.csv')
scenario_metrics = pd.read_csv(ROOT/'scenario_study/results/metrics_by_scenario_metal.csv')
scenario_qc = pd.read_csv(ROOT/'scenario_study/results/acceptance_by_scenario.csv')
assert len(baseline_labels)==500 and len(expanded_labels)==1800
assert baseline_labels.groupby('session_id').partition.nunique().max()==1
assert expanded_labels.groupby('session_id').partition.nunique().max()==1
assert expanded_labels.groupby('sample_id').partition.nunique().max()==1
print('Original:', len(baseline_labels), 'images;', baseline_labels.sample_id.nunique(), 'profiles')
print('Expanded:', len(expanded_labels), 'images;', expanded_labels.sample_id.nunique(), 'profiles')
'''),
    md('## 7. Compare model results without calling them accuracy\nMAE/RMSE are simulated concentration errors. R² describes regression fit, not percent correct. NRMSE is scaled by each metal’s assumed full-scale concentration. The shifted-camera challenge is a simulation, not a physical validation.'),
    code('''from IPython.display import display, Markdown, Image
display(Markdown(build_research_interpretation()))
display(Markdown((ROOT/'scenario_study/REPORT.md').read_text(encoding='utf-8')))
display(scenario_metrics)
display(scenario_qc)
display(Image(filename=str(ROOT/'charts/model_comparison.png')))
display(Image(filename=str(ROOT/'scenario_study/scenario_model_comparison.png')))
display(Image(filename=str(ROOT/'scenario_study/image_acceptance.png')))
display(Markdown((V2_DIR/'REPORT.md').read_text(encoding='utf-8')))
display(pd.read_csv(V2_DIR/'coverage_and_acceptance.csv'))
display(Image(filename=str(V2_DIR/'comparison.png')))
'''),
    md('## 8. Predict one included synthetic image\nThis is a workflow demonstration only. Rejected captures yield no concentrations. A real strip image is not supported without physical calibration and laboratory reference measurements.'),
    code('''example = baseline_labels[baseline_labels.partition=='test'].iloc[0]
result, corrected = estimate_improved(ROOT/example.image_path)
result['interpretation'] = interpret_prediction(result)
print('Image:', example.image_id)
print('Interpretation:', result['interpretation'])
display(pd.DataFrame(result['predictions']))
'''),
    md('''## 9. Research boundary

The first 500 and the new 1,800 images are separate synthetic experiments, not 2,300 independent real samples. All concentrations originate from assumed color-generating equations. The expanded study’s intentional corruptions add 120 images but **zero independent profiles**. For real-world claims, manufacture the seven-metal strip, standardize ESP32-S3 capture conditions, collect multiple physical lots and water matrices, pair images with reference-laboratory concentrations, and reserve a fully independent external test campaign.'''),
]

notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
})
nbformat.validate(notebook)
notebook_path = ROOT / 'AquaSense_All_In_One.ipynb'
nbformat.write(notebook, notebook_path)

# Execute all cells in one fresh namespace without retraining on the default path.
namespace = {'__name__': '__notebook__'}
for i, cell in enumerate(cells):
    if cell.cell_type == 'code':
        exec(compile(cell.source, f'unified_notebook_cell_{i}', 'exec'), namespace)
assert len(namespace['result']['predictions']) == 7
print('Unified notebook default path executed successfully.')

included = []
for path in ROOT.rglob('*'):
    if not path.is_file():
        continue
    relative = path.relative_to(ROOT)
    if any(part in {'__pycache__', 'runtime'} for part in relative.parts):
        continue
    if path.suffix in {'.pyc', '.tmp'}:
        continue
    if path.suffix == '.ipynb' and path != notebook_path:
        continue
    if path.name.startswith('package_') and path.suffix == '.py':
        continue
    included.append(path)
with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for path in sorted(included):
        bundle.write(path, Path('AquaSense') / path.relative_to(ROOT))
with zipfile.ZipFile(ARCHIVE) as bundle:
    assert bundle.testzip() is None
    names = bundle.namelist()
    assert sum(name.endswith('.ipynb') for name in names) == 1
    assert sum(name.startswith('AquaSense/datasets/images/') and name.endswith('.jpg') for name in names) == 500
    assert sum(name.startswith('AquaSense/scenario_study/images/') and name.endswith('.jpg') for name in names) == 1920
    assert any(name == 'AquaSense/AquaSense_All_In_One.ipynb' for name in names)
print(f'One notebook + complete ZIP verified: {ARCHIVE} ({ARCHIVE.stat().st_size/1048576:.1f} MB)')
