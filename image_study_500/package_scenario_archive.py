"""Bundle the self-contained expanded synthetic scenario study."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / 'AquaSense_Expanded_Synthetic_Scenarios.zip'
files = [ROOT / name for name in [
    'AquaSense_Scenario_Study.ipynb', 'requirements.txt', 'scenario_study.py',
    'image_pipeline.py', 'quality_control.py', 'models/reliable/config.json',
]]
files += [p for p in (ROOT / 'scenario_study').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
missing = [str(p) for p in files if not p.is_file()]
if missing:
    raise FileNotFoundError(f'Missing bundle files: {missing}')
with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for path in sorted(set(files)):
        bundle.write(path, path.relative_to(ROOT.parent))
with zipfile.ZipFile(ARCHIVE) as bundle:
    assert bundle.testzip() is None
    names = bundle.namelist()
    assert sum('/scenario_study/images/' in name and name.endswith('.jpg') for name in names) == 1920
    assert sum(name.endswith('.ipynb') for name in names) == 1
    assert any(name.endswith('/scenario_study/REPORT.md') for name in names)
print(f'Verified {ARCHIVE} ({ARCHIVE.stat().st_size / 1048576:.1f} MB)')
