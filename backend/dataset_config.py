"""
Breast Cancer Dataset Configuration

This file contains information about available breast cancer datasets
and how to access them.
"""

# Dataset configurations
BREAST_CANCER_DATASETS = {
    "breakhis": {
        "name": "BreakHis - Breast Cancer Histopathological Image Classification",
        "url": "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
        "description": "7,909 microscopic images of breast tumor tissue collected from 82 patients",
        "classes": {
            "benign": ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"],
            "malignant": ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]
        },
        "magnifications": ["40X", "100X", "200X", "400X"],
        "format": "PNG",
        "size": "1.6 GB",
        "access": "Free with registration",
        "filename": "BreaKHis_v1.tar.gz"
    },
    
    "bach": {
        "name": "BACH - BreAst Cancer Histology Challenge",
        "url": "https://iciar2018-challenge.grand-challenge.org/dataset/",
        "description": "ICIAR 2018 challenge dataset for breast cancer histology classification",
        "classes": {
            "normal": "Normal tissue",
            "benign": "Benign lesion", 
            "in_situ": "In situ carcinoma",
            "invasive": "Invasive carcinoma"
        },
        "format": "TIFF",
        "size": "~400 images for training",
        "access": "Free with registration"
    },
    
    "kaggle_idc": {
        "name": "Invasive Ductal Carcinoma (IDC) Detection",
        "url": "https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images",
        "description": "277,524 patches of size 50x50 extracted from 162 WSI breast cancer slides",
        "classes": {
            "negative": "No IDC",
            "positive": "IDC detected"
        },
        "kaggle_command": "kaggle datasets download -d paultimothymooney/breast-histopathology-images",
        "format": "PNG",
        "size": "1.2 GB",
        "access": "Kaggle account required"
    },
    
    "camelyon16": {
        "name": "Camelyon16 - Metastases Detection",
        "url": "https://camelyon16.grand-challenge.org/",
        "description": "Lymph node metastases detection in breast cancer",
        "format": "WSI (Whole Slide Images)",
        "size": "Large dataset",
        "access": "Free with registration"
    }
}

# Recommended dataset based on use case
RECOMMENDED_DATASETS = {
    "beginner": "kaggle_idc",  # Easiest to download and use
    "research": "breakhis",    # Most comprehensive
    "challenge": "bach",       # Competition standard
    "advanced": "camelyon16"   # Most realistic clinical data
}

# Quick start instructions
QUICK_START_GUIDE = """
🎗️ Quick Start Guide for Real Breast Cancer Data:

1. EASIEST (Kaggle IDC Dataset):
   - Install: pip install kaggle
   - Setup Kaggle API credentials
   - Run: kaggle datasets download -d paultimothymooney/breast-histopathology-images
   - Extract to backend/data/idc/

2. MOST COMPREHENSIVE (BreakHis):
   - Visit: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
   - Fill registration form
   - Download BreaKHis_v1.tar.gz
   - Place in backend/data/
   - Your system will auto-extract it

3. RESEARCH STANDARD (BACH):
   - Visit: https://iciar2018-challenge.grand-challenge.org/dataset/
   - Register and download
   - Extract to backend/data/bach/

Then restart your backend server - it will automatically use real data!
"""