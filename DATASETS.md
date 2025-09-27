# Mammography Datasets You Can Use

Below are well-known, high-quality datasets to strengthen training and improve malignant detection. They vary in size, format, and access requirements.

## 1) RSNA Screening Mammography (Kaggle)

- URL: https://www.kaggle.com/competitions/rsna-breast-cancer-detection
- Size: Very large (hundreds of thousands of images)
- Format: DICOM, with patient-level cancer labels. A subset has bounding boxes.
- Pros: Modern FFDM, scale for deep learning, diverse scanners.
- Cons: Labels are mostly patient-level; requires careful split by patient; Kaggle download.
- Notes: Great for pretraining and representation learning; consider pseudo-labeling or MIL.

## 2) CBIS-DDSM (Curated DDSM)

- URL: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
- Format: DICOM (curated subset of DDSM)
- Pros: Pathology-proven labels; curated; train/test splits; metadata.
- Cons: Requires free TCIA account; mid-size.

## 3) VinDr-Mammo

- URL: https://physionet.org/content/vindr-mammo/1.0.0/
- Format: DICOM/PNG + CSV labels; BI-RADS assessments; bounding boxes for findings
- Pros: Detailed annotations, modern images, free with credentialed access.
- Cons: Preprocessing and label mapping needed.

## 4) CMMD (Chinese Mammography Database)

- URL: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508
- Format: DICOM + clinical metadata
- Pros: Pathology-confirmed; metadata; complementary population.
- Cons: TCIA download; structure varies.

## 5) INbreast

- URL: http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database
- Format: DICOM
- Pros: High-quality FFDM; widely used for research.
- Cons: Access requires application; smaller than RSNA.

## 6) Original DDSM

- URL: http://www.eng.usf.edu/cvprg/Mammography/Database.html
- Format: LJPEG (needs conversion) or converted images
- Pros: Classic large dataset; many works use it.
- Cons: Legacy format; preprocessing heavy.

## 7) BCDR (BCDR-DM / BCDR-FM)

- URL: http://bcdr.inegi.up.pt/
- Format: PNG/JPG/DICOM with ROI annotations
- Pros: Annotated lesions and metadata.
- Cons: Access by request; smaller than RSNA.

## 8) MIAS (current baseline)

- URL: http://peipa.essex.ac.uk/info/mias.html
- Format: PGM (can convert to PNG)
- Pros: Easy to start; small footprint.
- Cons: Small and older; limited for deep models.

---

## Recommended Strategy

- Pretrain or co-train on a large dataset (RSNA or VinDr-Mammo), then fine-tune on CBIS-DDSM/INbreast/MIAS.
- Use patient-wise splits to avoid leakage.
- Harmonize labels to a binary task: {benign, malignant}; optionally include "normal" as benign.
- Convert DICOM to PNG for faster experimentation (preserve windowing). Keep DICOMs for final training.
- Use bounding boxes (RSNA subset, VinDr-Mammo) for auxiliary detection/localization loss or ROI cropping.

## Wiring Into This Repo

- Place datasets under `backend/data/<dataset_name>` or anywhere you prefer.
- Run:

```fish
python3 backend/dataset_utils.py
```

- It will print which datasets are detected locally and choose the preferred one automatically.

## Troubleshooting

- If a dataset isn't detected, adjust `DATASET_SEARCH_PATHS` in `backend/dataset_utils.py` or set an env var in your training script to point to the dataset root.
- DICOM reading requires `pydicom`, `pylibjpeg`, `gdcm` (optional). Ensure your `requirements.txt` includes them when you start using DICOM.

---

## Handling Very Large Datasets (RSNA ~350GB)

You don’t need to store raw datasets in Git. Use one of these patterns instead:

1. Keep data out of Git (recommended)

   - We already ignore `backend/data/` in `.gitignore` so local datasets won’t be committed.
   - Put datasets under `backend/data/` or mount them via Docker volumes.

2. Download subsets only (space-savvy)

   - Prefer smaller, high-signal subsets:
     - VinDr-Mammo (PhysioNet) – moderate size, rich labels.
     - CBIS-DDSM (TCIA) – curated and smaller than RSNA.
   - For RSNA, download a limited number of exams for experiments first.

3. Convert to lighter formats and crop ROIs

   - Convert DICOM → 8-bit PNG with robust windowing (we provide `setup_vindr_mammo.py`).
   - Crop around bounding boxes (when available) to reduce storage.
   - Optionally downscale to 1024–1536px on the long side for prototyping.

4. Stream or mount storage instead of copying
   - Mount an external drive or network path into `backend/data`.
   - Use Docker bind mounts in `docker-compose.yml`:

```yaml
volumes:
  - /path/to/big/drive/datasets:/app/backend/data:ro
```

5. Use a data registry rather than Git

   - DVC (Data Version Control) or Git LFS for pointers; store blobs in S3/GCS/Drive.
   - Keep only small samples in Git; pull full data on-demand.

6. On-demand conversion scripts (no storage duplication)
   - Use `backend/setup_vindr_mammo.py` to directly convert only N images:

```fish
python3 backend/setup_vindr_mammo.py \
  --vindr-root "/mnt/datasets/vindr-mammo" \
  --output-dir "backend/data/mammography" \
  --birads3 exclude \
  --max-images 2000
```

7. Train with patient-wise sampling over a mounted dataset
   - Change the training script to read from the mounted path and iterate without copying all files.
   - Keep a CSV manifest (image path, label, study_id) to control splits and sampling.

Quick sanity checks

```fish
# Verify the repo ignores large data
git check-ignore -v backend/data/**

# Detect available datasets without downloading
python3 backend/dataset_utils.py
```
