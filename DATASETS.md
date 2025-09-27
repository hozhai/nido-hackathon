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
