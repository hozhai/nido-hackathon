"""
Dataset Configuration for Breast Cancer Detection using Mammography
This file contains configuration settings for different mammography datasets
"""

import os

# Dataset types
DATASET_SYNTHETIC = "synthetic"
DATASET_CBIS_DDSM = "cbis_ddsm"
DATASET_DDSM = "ddsm"
DATASET_INBREAST = "inbreast"
DATASET_MIAS = "mias"
DATASET_RSNA = "rsna_bcd"
DATASET_VINDR = "vindr_mammo"
DATASET_CMMD = "cmmd"
DATASET_BCDR = "bcdr"

# Default dataset preference order (from most to least preferred if available)
DATASET_PREFERENCE = [
    DATASET_RSNA,       # RSNA Screening Mammography (large, modern DICOM, Kaggle)
    DATASET_CBIS_DDSM,  # Curated DDSM subset (TCIA)
    DATASET_VINDR,      # VinDr-Mammo (PhysioNet) with BI-RADS and bboxes
    DATASET_CMMD,       # Chinese Mammography Database (TCIA)
    DATASET_INBREAST,   # High-quality full-field DICOM
    DATASET_DDSM,       # Original legacy DDSM (LJPEG)
    DATASET_BCDR,       # BCDR mammography (by request)
    DATASET_MIAS,       # Classic small dataset
    DATASET_SYNTHETIC   # Fallback to synthetic
]

# Dataset configurations
DATASET_CONFIG = {
    DATASET_RSNA: {
        "name": "RSNA Screening Mammography Breast Cancer Detection (Kaggle)",
        "description": "Large-scale screening mammography dataset with patient-level cancer labels; subset includes bounding boxes for cancers.",
        "classes": ["No Cancer", "Cancer"],
        "binary_mapping": {
            "No Cancer": "benign",
            "Cancer": "malignant"
        },
        "expected_structure": "rsna-bcd/[train_images|test_images]/<study_id>/<image>.dcm",
        "image_extensions": [".dcm"],
        "typical_image_size": (3000, 4000),
        "url": "https://www.kaggle.com/competitions/rsna-breast-cancer-detection",
        "notes": "Requires Kaggle account; labels are mostly patient-level. Bounding boxes available for positive cases."
    },

    DATASET_CBIS_DDSM: {
        "name": "CBIS-DDSM (Curated Digital Database for Screening Mammography)",
        "description": "Curated mammography dataset with pathology-proven labels",
        "classes": ["Normal", "Benign", "Malignant"],
        "binary_mapping": {
            "Normal": "benign",
            "Benign": "benign",
            "Malignant": "malignant"
        },
        "expected_structure": "CBIS-DDSM/[Mass|Calc]/[Train|Test]/[Benign|Malignant]/",
        "image_extensions": [".dcm", ".png", ".jpg"],
        "typical_image_size": (3328, 4096),  # Full-field digital mammography
        "url": "https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM",
        "notes": "High-quality DICOM format, requires TCIA account (free)"
    },
    
    DATASET_VINDR: {
        "name": "VinDr-Mammo",
        "description": "Large-scale screening mammography dataset with BI-RADS assessments and bounding boxes for findings.",
        "classes": ["Normal", "Benign", "Malignant"],
        "binary_mapping": {
            "Normal": "benign",
            "Benign": "benign",
            "Malignant": "malignant"
        },
        "expected_structure": "vindr-mammo/[images|labels]/",
        "image_extensions": [".dcm", ".png", ".jpg"],
        "typical_image_size": (3000, 4000),
        "url": "https://physionet.org/content/vindr-mammo/1.0.0/",
        "notes": "Free for research via PhysioNet credentialed access; includes bounding boxes and BI-RADS."
    },

    DATASET_CMMD: {
        "name": "CMMD (Chinese Mammography Database)",
        "description": "Mammography dataset with pathology-confirmed labels and metadata.",
        "classes": ["Normal", "Benign", "Malignant"],
        "binary_mapping": {
            "Normal": "benign",
            "Benign": "benign",
            "Malignant": "malignant"
        },
        "expected_structure": "CMMD/[full_dataset]/<patient>/<image>.dcm",
        "image_extensions": [".dcm"],
        "typical_image_size": (2500, 3000),
        "url": "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508",
        "notes": "Available on TCIA; includes clinical metadata."
    },

    DATASET_DDSM: {
        "name": "DDSM (Digital Database for Screening Mammography)",
        "description": "Original large mammography database",
        "classes": ["Normal", "Benign", "Malignant"],
        "binary_mapping": {
            "Normal": "benign",
            "Benign": "benign", 
            "Malignant": "malignant"
        },
        "expected_structure": "DDSM/[case_type]/case_*/",
        "image_extensions": [".ljpeg", ".png", ".jpg"],
        "typical_image_size": (4096, 6144),
        "url": "http://www.eng.usf.edu/cvprg/Mammography/Database.html",
        "notes": "Large dataset, LJPEG format requires conversion"
    },
    
    DATASET_INBREAST: {
        "name": "INbreast Database",
        "description": "Full-field digital mammography database",
        "classes": ["Normal", "Benign", "Malignant"],
        "binary_mapping": {
            "Normal": "benign",
            "Benign": "benign",
            "Malignant": "malignant"
        },
        "expected_structure": "INbreast/AllDICOMs/",
        "image_extensions": [".dcm"],
        "typical_image_size": (3328, 2560),
        "url": "http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database",
        "notes": "High-quality DICOM, requires research application"
    },

    DATASET_BCDR: {
        "name": "BCDR (Breast Cancer Digital Repository) - Mammography",
        "description": "Mammography datasets with ROI annotations (e.g., BCDR-DM).",
        "classes": ["Normal", "Benign", "Malignant"],
        "binary_mapping": {
            "Normal": "benign",
            "Benign": "benign",
            "Malignant": "malignant"
        },
        "expected_structure": "BCDR/[BCDR-DM|BCDR-FM]/",
        "image_extensions": [".png", ".jpg", ".dcm"],
        "typical_image_size": (3000, 4000),
        "url": "http://bcdr.inegi.up.pt/",
        "notes": "Access by request; includes ROI-level annotations."
    },
    
    DATASET_MIAS: {
        "name": "MIAS (Mammographic Image Analysis Society)",
        "description": "Classic mammography dataset",
        "classes": ["Normal", "Benign", "Malignant"],
        "binary_mapping": {
            "Normal": "benign",
            "Benign": "benign",
            "Malignant": "malignant"
        },
        "expected_structure": "MIAS/[all_images]/",
        "image_extensions": [".pgm", ".png", ".jpg"],
        "typical_image_size": (1024, 1024),
        "url": "http://peipa.essex.ac.uk/info/mias.html", 
        "notes": "Smaller dataset, PGM format, good for testing"
    },
    
    DATASET_SYNTHETIC: {
        "name": "Synthetic Mammography Data",
        "description": "Automatically generated synthetic mammography images",
        "classes": ["benign", "malignant"],
        "binary_mapping": {
            "benign": "benign",
            "malignant": "malignant"
        },
        "image_extensions": [".png"],
        "typical_image_size": (512, 512),
        "notes": "Generated automatically, not suitable for real medical use"
    }
}

# Search paths for datasets
DATASET_SEARCH_PATHS = {
    DATASET_RSNA: [
        "./RSNA", "./rsna", "./rsna-bcd", "./data/RSNA", "./data/rsna-bcd", "./data/rsna",
        "../RSNA", "../rsna-bcd",
        os.path.expanduser("~/Downloads/RSNA"), os.path.expanduser("~/Downloads/rsna-bcd"),
        os.path.expanduser("~/Documents/RSNA"), os.path.expanduser("~/Documents/rsna-bcd"),
        "/tmp/RSNA", "/tmp/rsna-bcd",
    ],

    DATASET_CBIS_DDSM: [
        "./CBIS-DDSM",
        "./cbis_ddsm",
        "./data/CBIS-DDSM",
        "./data/cbis_ddsm",
        "../CBIS-DDSM",
        "../cbis_ddsm",
        os.path.expanduser("~/Downloads/CBIS-DDSM"),
        os.path.expanduser("~/Downloads/cbis_ddsm"),
        os.path.expanduser("~/Documents/CBIS-DDSM"),
        os.path.expanduser("~/Documents/cbis_ddsm"),
        "/tmp/CBIS-DDSM",
        "/tmp/cbis_ddsm",
    ],

    DATASET_VINDR: [
        "./VinDr-Mammo", "./vindr-mammo", "./vindr_mammo", "./data/vindr-mammo", "./data/vindr_mammo",
        "../VinDr-Mammo", "../vindr-mammo",
        os.path.expanduser("~/Downloads/VinDr-Mammo"), os.path.expanduser("~/Downloads/vindr-mammo"),
        os.path.expanduser("~/Documents/VinDr-Mammo"),
        "/tmp/VinDr-Mammo",
    ],
    
    DATASET_DDSM: [
        "./DDSM",
        "./ddsm",
        "./data/DDSM", 
        "./data/ddsm",
        "../DDSM",
        "../ddsm",
        os.path.expanduser("~/Downloads/DDSM"),
        os.path.expanduser("~/Downloads/ddsm"),
        os.path.expanduser("~/Documents/DDSM"),
        "/tmp/DDSM",
    ],

    DATASET_CMMD: [
        "./CMMD", "./data/CMMD", "../CMMD",
        os.path.expanduser("~/Downloads/CMMD"), os.path.expanduser("~/Documents/CMMD"),
        "/tmp/CMMD",
    ],
    
    DATASET_INBREAST: [
        "./INbreast",
        "./inbreast",
        "./data/INbreast",
        "./data/inbreast", 
        "../INbreast",
        "../inbreast",
        os.path.expanduser("~/Downloads/INbreast"),
        os.path.expanduser("~/Downloads/inbreast"),
        os.path.expanduser("~/Documents/INbreast"),
        "/tmp/INbreast",
    ],
    
    DATASET_MIAS: [
        "./MIAS",
        "./mias",
        "./data/MIAS",
        "./data/mias",
        "../MIAS", 
        "../mias",
        os.path.expanduser("~/Downloads/MIAS"),
        os.path.expanduser("~/Downloads/mias"),
        os.path.expanduser("~/Documents/MIAS"),
        "/tmp/MIAS",
    ]
}

def get_dataset_info(dataset_type):
    """Get configuration information for a specific dataset type"""
    return DATASET_CONFIG.get(dataset_type, {})

def find_available_datasets():
    """Find which datasets are available on the system"""
    available = []
    
    # Check RSNA (Kaggle) - typically has train_images with many DICOMs
    for path in DATASET_SEARCH_PATHS.get(DATASET_RSNA, []):
        if os.path.exists(path):
            candidates = [
                os.path.join(path, "train_images"),
                os.path.join(path, "images"),
            ]
            if any(os.path.isdir(p) and any(fn.endswith('.dcm') for fn in os.listdir(p)) for p in candidates if os.path.exists(p)):
                available.append((DATASET_RSNA, path))
                break

    # Check CBIS-DDSM
    for path in DATASET_SEARCH_PATHS.get(DATASET_CBIS_DDSM, []):
        if os.path.exists(path):
            # Check for CBIS-DDSM structure (Mass or Calc directories)
            if (os.path.exists(os.path.join(path, "Mass")) or 
                os.path.exists(os.path.join(path, "Calc")) or
                any(os.path.exists(os.path.join(path, subdir)) 
                    for subdir in ["Train", "Test", "CBIS-DDSM"])):
                available.append((DATASET_CBIS_DDSM, path))
                break
    
    # Check VinDr-Mammo
    for path in DATASET_SEARCH_PATHS.get(DATASET_VINDR, []):
        if os.path.exists(path):
            # Look for images directory with DICOM/PNG/JPG or presence of annotation files
            images_dir = None
            for candidate in ["images", "pngs", "imgs"]:
                cand_path = os.path.join(path, candidate)
                if os.path.isdir(cand_path):
                    images_dir = cand_path
                    break
            has_images = False
            if images_dir:
                try:
                    has_images = any(
                        f.lower().endswith((".dcm", ".png", ".jpg")) for f in os.listdir(images_dir)
                    )
                except Exception:
                    has_images = False
            ann_files = [f for f in os.listdir(path) if f.lower().endswith((".csv", ".json"))]
            if has_images or ann_files:
                available.append((DATASET_VINDR, path))
                break

    # Check DDSM
    for path in DATASET_SEARCH_PATHS.get(DATASET_DDSM, []):
        if os.path.exists(path):
            # Check for DDSM structure (case directories)
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if any("case" in d.lower() for d in subdirs) or len(subdirs) > 100:
                available.append((DATASET_DDSM, path))
                break
    
    # Check CMMD
    for path in DATASET_SEARCH_PATHS.get(DATASET_CMMD, []):
        if os.path.exists(path):
            # DICOM presence or metadata files
            has_dcm = any(
                f.lower().endswith('.dcm') for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
            )
            has_meta = any(
                f.lower() in ("metadata.csv", "manifest.csv") for f in os.listdir(path)
            )
            if has_dcm or has_meta:
                available.append((DATASET_CMMD, path))
                break

    # Check INbreast
    for path in DATASET_SEARCH_PATHS.get(DATASET_INBREAST, []):
        if os.path.exists(path):
            # Check for INbreast structure (DICOM files or AllDICOMs directory)
            all_dicoms_dir = os.path.join(path, "AllDICOMs")
            if (os.path.exists(all_dicoms_dir) or 
                any(f.endswith('.dcm') for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))):
                available.append((DATASET_INBREAST, path))
                break
    
    # Check BCDR (any subfolder containing mammography images)
    for path in DATASET_SEARCH_PATHS.get(DATASET_BCDR, []):
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            has_imgs = any(f.lower().endswith(('.png', '.jpg', '.dcm')) for f in files)
            if has_imgs or any(s for s in os.listdir(path) if "BCDR" in s):
                available.append((DATASET_BCDR, path))
                break

    # Check MIAS
    for path in DATASET_SEARCH_PATHS.get(DATASET_MIAS, []):
        if os.path.exists(path):
            # Check for MIAS structure (PGM files or converted images)
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            if (any(f.endswith('.pgm') for f in files) or
                any(f.startswith('mdb') for f in files)):  # MIAS naming convention
                available.append((DATASET_MIAS, path))
                break
    
    # Synthetic is always available
    available.append((DATASET_SYNTHETIC, "generated"))
    
    return available

def get_preferred_dataset():
    """Get the most preferred available dataset"""
    available = find_available_datasets()
    available_types = [dataset_type for dataset_type, _ in available]
    
    for preferred in DATASET_PREFERENCE:
        if preferred in available_types:
            # Return the first match with its path
            for dataset_type, path in available:
                if dataset_type == preferred:
                    return dataset_type, path
    
    # Fallback
    return DATASET_SYNTHETIC, "generated"

def print_dataset_status():
    """Print status of all datasets"""
    print("🔬 Dataset Availability Status:")
    print("-" * 40)
    
    available = find_available_datasets()
    available_dict = dict(available)
    
    for dataset_type in DATASET_PREFERENCE:
        config = get_dataset_info(dataset_type)
        name = config.get("name", dataset_type)
        
        if dataset_type in available_dict:
            path = available_dict[dataset_type]
            status = f"✅ Available at: {path}"
        else:
            status = "❌ Not found"
        
        print(f"{name}:")
        print(f"  {status}")
        
        if config.get("url"):
            print(f"  🌐 URL: {config['url']}")
        
        if config.get("notes"):
            print(f"  📝 {config['notes']}")
        
        print()

if __name__ == "__main__":
    print_dataset_status()
    
    preferred_type, preferred_path = get_preferred_dataset()
    print(f"🎯 Preferred dataset: {preferred_type} at {preferred_path}")