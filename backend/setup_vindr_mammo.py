#!/usr/bin/env python3
"""
Prepare VinDr-Mammo (PhysioNet) dataset for binary classification (benign vs malignant).

Mapping strategy:
- Use breast-level BI-RADS per image (breast_birads) as primary label.
  * BI-RADS 1-2 -> benign
  * BI-RADS 4-5-6 -> malignant
  * BI-RADS 0 -> exclude (incomplete)
  * BI-RADS 3 -> exclude by default (option to include as benign)

- If finding_annotations.csv is present, upgrade to malignant if any finding on the image
  has finding_birads >= 4.

Output:
- Converted PNG images under data/mammography/{benign,malignant}
- Ensures patient/study leakage is avoidable in training by using study_id-aware splits (handled elsewhere).

Usage (fish shell):
  python3 backend/setup_vindr_mammo.py --vindr-root \
    "/path/to/vindr-mammo" --output-dir \
    "/home/zhai/Documents/nido-hackathon/backend/data/mammography" --birads3 exclude --max-images 5000
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except Exception as e:
    pydicom = None

from PIL import Image


def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _read_csv_index(csv_path: Path) -> Tuple[Dict[str, dict], list]:
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    index = {}
    for r in rows:
        # Key per image
        image_id = r.get('image_id') or r.get('filename') or ''
        key = image_id.strip()
        if key:
            index[key] = r
    return index, rows


def _load_finding_birads_map(finding_csv: Optional[Path]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not finding_csv or not finding_csv.exists():
        return mapping
    with open(finding_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            image_id = (r.get('image_id') or '').strip()
            try:
                fb = r.get('finding_birads')
                if fb is None or fb == '':
                    continue
                fb_int = int(float(fb))
            except Exception:
                continue
            # Keep the max finding BI-RADS per image
            mapping[image_id] = max(mapping.get(image_id, -1), fb_int)
    return mapping


def birads_to_binary(breast_birads: Optional[str], finding_birads: Optional[int], birads3_policy: str) -> Optional[int]:
    # Returns 0 (benign), 1 (malignant), or None for exclude
    val = None
    try:
        if breast_birads is None or breast_birads == '':
            val = None
        else:
            b = int(float(breast_birads))
            if b in (1, 2):
                val = 0
            elif b in (4, 5, 6):
                val = 1
            elif b == 3:
                if birads3_policy == 'benign':
                    val = 0
                else:
                    val = None
            elif b == 0:
                val = None
            else:
                val = None
    except Exception:
        val = None

    # Upgrade to malignant if any finding BI-RADS >= 4
    if finding_birads is not None and finding_birads >= 4:
        return 1
    return val


def dicom_to_uint8_png(dcm_path: Path) -> Image.Image:
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array
    # Apply VOI LUT if present
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    # Convert to float
    arr = arr.astype(np.float32)

    # Invert if MONOCHROME1 (where higher values are darker)
    photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    if photometric == 'MONOCHROME1':
        arr = np.max(arr) - arr

    # Robust windowing via percentiles
    p_low, p_high = np.percentile(arr, [0.5, 99.5])
    if p_high <= p_low:
        p_low, p_high = arr.min(), arr.max()
    arr = np.clip((arr - p_low) / (p_high - p_low + 1e-6), 0, 1)
    arr = (arr * 255.0).round().astype(np.uint8)

    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser(description='Prepare VinDr-Mammo (PhysioNet) into benign/malignant PNGs')
    parser.add_argument('--vindr-root', required=True, help='Path to VinDr-Mammo root with images/ and CSVs')
    parser.add_argument('--output-dir', default=str(Path(__file__).parent / 'data' / 'mammography'), help='Output dir for benign/malignant folders')
    parser.add_argument('--birads3', choices=['exclude', 'benign'], default='exclude', help='How to treat BI-RADS 3 (default: exclude)')
    parser.add_argument('--max-images', type=int, default=None, help='Optional cap on processed images for a quick start')
    parser.add_argument('--copy-dicom', action='store_true', help='Also copy original DICOM into class folder (besides PNG)')

    args = parser.parse_args()

    if pydicom is None:
        print('❌ Missing dependency: pydicom. Please add pydicom and pylibjpeg-libjpeg to requirements and install.')
        sys.exit(1)

    vindr_root = Path(args.vindr_root)
    images_root = vindr_root / 'images'
    breast_csv = vindr_root / 'breast-level_annotations.csv'
    finding_csv = vindr_root / 'finding_annotations.csv'

    if not images_root.exists() or not breast_csv.exists():
        print('❌ Could not find VinDr-Mammo structure. Expecting images/ and breast-level_annotations.csv')
        sys.exit(1)

    # Read CSVs
    breast_index, breast_rows = _read_csv_index(breast_csv)
    finding_map = _load_finding_birads_map(finding_csv if finding_csv.exists() else None)

    out_dir = Path(args.output_dir)
    benign_dir = out_dir / 'benign'
    malignant_dir = out_dir / 'malignant'
    _safe_mkdir(benign_dir)
    _safe_mkdir(malignant_dir)

    processed = 0
    kept = 0
    benign_count = 0
    malignant_count = 0

    print('🔧 Converting VinDr-Mammo DICOMs to PNG and mapping BI-RADS → binary...')

    # Iterate breast-level rows (each corresponds to an image)
    for r in breast_rows:
        image_id = (r.get('image_id') or '').strip()
        study_id = (r.get('study_id') or '').strip()
        breast_birads = r.get('breast_birads')

        if not image_id or not study_id:
            continue

        # Path to DICOM: images/<study_id>/<image_id>.dicom
        dcm_path = images_root / study_id / f'{image_id}.dicom'
        if not dcm_path.exists():
            continue

        finding_birads = finding_map.get(image_id)
        label = birads_to_binary(breast_birads, finding_birads, args.birads3)
        if label is None:
            # excluded cases (BI-RADS 0 or 3 by default)
            continue

        try:
            img = dicom_to_uint8_png(dcm_path)
        except Exception as e:
            # Skip unreadable DICOMs
            continue

        target_dir = benign_dir if label == 0 else malignant_dir
        target_name = f'vindr_{study_id}_{image_id}.png'
        target_path = target_dir / target_name
        try:
            img.save(target_path)
            if args.copy_dicom:
                import shutil
                shutil.copy2(dcm_path, target_dir / f'vindr_{study_id}_{image_id}.dicom')
        except Exception:
            continue

        kept += 1
        if label == 0:
            benign_count += 1
        else:
            malignant_count += 1

        processed += 1
        if args.max_images is not None and processed >= args.max_images:
            break

    print(f'✅ VinDr-Mammo prepared at: {out_dir}')
    print(f'   Benign: {benign_count}, Malignant: {malignant_count}, Total: {kept}')
    if benign_count == 0 and malignant_count == 0:
        print('⚠️ No images were prepared. Check BI-RADS mapping or CSV structure.')

if __name__ == '__main__':
    main()
