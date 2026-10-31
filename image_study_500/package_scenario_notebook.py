"""Build a self-contained notebook for the expanded synthetic scenario study."""
from pathlib import Path
import ast
import nbformat

ROOT = Path(__file__).resolve().parent


def module_source(name, remove):
    source = (ROOT / name).read_text(encoding='utf-8')
    source = source[:source.index("if __name__ == '__main__':") if "if __name__ == '__main__':" in source else source.index("if __name__=='__main__':")]
    for line in remove:
        source = source.replace(line, '')
    ast.parse(source)
    return source


base = module_source('image_pipeline.py', ['ROOT = Path(__file__).resolve().parent'])
qc = module_source('quality_control.py', ['from image_pipeline import ROOT, SPOTS, REF_CENTERS, preprocess_image\n']) if "if __name__" in (ROOT / 'quality_control.py').read_text(encoding='utf-8') else (ROOT / 'quality_control.py').read_text(encoding='utf-8').replace('from image_pipeline import ROOT, SPOTS, REF_CENTERS, preprocess_image\n', '')
scenario = module_source('scenario_study.py', [
    'from image_pipeline import ROOT, METALS, TARGETS, MAXIMA, FEATURES, assay_colors, render_card, camera_capture, model_spaces, preprocess_image\n',
    'from quality_control import ImageRejected, check_image\n',
])
cells = [
    nbformat.v4.new_markdown_cell('''# AquaSense expanded synthetic scenarios

**Simulation only:** 1,800 primary images from 900 new simulated profiles, plus 120 damaged-image QC cases. No real chemistry, manufactured strip, or ESP32 capture is represented.

Run this notebook from its folder. It contains the renderer, preprocessing, QC, four model families, grouped evaluation, charts, and report code in one notebook. By default it reads the supplied completed results. Set `REBUILD = True` in the last cell to regenerate and retrain; that overwrites the expanded-study outputs and can take substantial time. The original 500-image study remains separate.

The same-scenario test and wholly unseen camera/batch-shift challenge are never used for tuning. Captures of the same profile and all profiles of one session stay in one partition.'''),
    nbformat.v4.new_markdown_cell('## 1. Imports, fixed seed, rendering and preprocessing'),
    nbformat.v4.new_code_cell('from pathlib import Path\nROOT = Path.cwd()\nassert (ROOT / "models/reliable/config.json").exists(), "Open the notebook in the project folder with the supplied QC config."\n'),
    nbformat.v4.new_code_cell(base),
    nbformat.v4.new_markdown_cell('## 2. Image-quality gate'),
    nbformat.v4.new_code_cell(qc),
    nbformat.v4.new_markdown_cell('## 3. Scenario generation, feature extraction, grouped model tuning and evaluation'),
    nbformat.v4.new_code_cell(scenario),
    nbformat.v4.new_markdown_cell('## 4. Read results or rebuild\nR² is regression fit, not detection accuracy. Mean NRMSE compares simulated full-scale errors across metals; inspect per-metal errors and image rejection rates too.'),
    nbformat.v4.new_code_cell('''REBUILD = False
if REBUILD:
    main()
from IPython.display import display, Markdown, Image
display(Markdown((OUT / 'REPORT.md').read_text(encoding='utf-8')))
display(pd.read_csv(RESULTS / 'metrics_by_scenario_metal.csv'))
display(pd.read_csv(RESULTS / 'acceptance_by_scenario.csv'))
display(Image(filename=str(OUT / 'scenario_model_comparison.png')))
display(Image(filename=str(OUT / 'image_acceptance.png')))
'''),
]
notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}})
nbformat.validate(notebook)
nbformat.write(notebook, ROOT / 'AquaSense_Scenario_Study.ipynb')
print('Wrote AquaSense_Scenario_Study.ipynb')
