"""Validate a real collection manifest without training on it or modifying it."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

METALS = ['Pb','Hg','Cd','As','Cr','Cu','Zn']
REQUIRED = ['image_id','sample_id','session_id','partition','image_path','collection_date','strip_lot','camera_board','camera_sensor','reference_method','reference_report_id','reaction_time_s','pH','temperature_C']+[f'{m}_ug_L' for m in METALS]

def validate_manifest(path):
    path=Path(path).resolve();frame=pd.read_csv(path);errors=[]
    missing=set(REQUIRED)-set(frame.columns)
    if missing: return [f'Missing columns: {sorted(missing)}']
    if frame.empty: return ['No measurements supplied. The blank template is not a dataset.']
    if frame[REQUIRED].isna().any().any(): errors.append('Required fields contain missing values; do not invent replacements.')
    if not frame.image_id.is_unique: errors.append('image_id must be unique.')
    if not frame.image_path.is_unique: errors.append('Image paths are duplicated.')
    if not set(frame.partition)<= {'train','calibration','test'}: errors.append('Unsupported partition names.')
    if set(frame.partition)!= {'train','calibration','test'}: errors.append('All three partitions are required before model validation.')
    for group in ['sample_id','session_id']:
        if (frame.groupby(group).partition.nunique()>1).any(): errors.append(f'{group} crosses partitions; this leaks related measurements.')
    for column in [f'{m}_ug_L' for m in METALS]+['reaction_time_s','pH','temperature_C']:
        values=pd.to_numeric(frame[column],errors='coerce')
        if not np.isfinite(values).all(): errors.append(f'{column} must contain numeric finite measurements; censored values need a separate protocol.')
        if column.endswith('_ug_L') or column=='reaction_time_s':
            if (values<0).any(): errors.append(f'{column} contains negative values.')
        if column.endswith('_ug_L') and (frame.assign(value=values).groupby('sample_id').value.nunique()>1).any():
            errors.append(f'{column} has inconsistent reference values for repeats of one sample; resolve provenance.')
    if pd.to_datetime(frame.collection_date,errors='coerce').isna().any(): errors.append('collection_date must be a valid date.')
    for row in frame.itertuples():
        image=(path.parent/str(row.image_path)).resolve()
        if not image.is_relative_to(path.parent): errors.append(f'{row.image_id}: image path escapes the collection folder.'); continue
        if not image.is_file(): errors.append(f'{row.image_id}: image file missing.'); continue
        img=cv2.imread(str(image))
        if img is None or img.shape[:2]!=(480,640): errors.append(f'{row.image_id}: capture must be a decodable 640x480 image.')
    return errors

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest');args=parser.parse_args()
    issues=validate_manifest(args.manifest)
    if issues:
        print('\n'.join(issues));raise SystemExit(1)
    print('Manifest structure passed. Laboratory validity and real model validation still require review.')
