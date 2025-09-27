#!/usr/bin/env python3
"""
Breast Cancer Dataset Download Helper

This script helps download and setup real breast cancer histopathology datasets
for training your breast cancer detection model.
"""

import os
import requests
import zipfile
import tarfile
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, filename: str) -> bool:
    """Download a file from URL"""
    try:
        logger.info(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {filename}: {str(e)}")
        return False

def setup_breakhis_dataset():
    """Instructions for downloading BreakHis dataset"""
    print("\n🔬 BreakHis Dataset Setup Instructions:")
    print("=" * 50)
    print("1. Visit: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/")
    print("2. Fill out the form to request access")
    print("3. Download 'BreaKHis_v1.tar.gz' (1.6 GB)")
    print("4. Place the file in your backend/data/ directory")
    print("5. Restart your backend server - it will automatically extract and organize the data")
    print("\n📊 Dataset Info:")
    print("   - 7,909 breast histopathology images")
    print("   - Benign: Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma")
    print("   - Malignant: Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma")
    print("   - Multiple magnification levels: 40X, 100X, 200X, 400X")

def download_bach_dataset():
    """Download BACH dataset from ICIAR 2018"""
    try:
        # BACH dataset URLs (if available)
        bach_urls = {
            "train": "https://iciar2018-challenge.grand-challenge.org/download/ICIAR2018_BACH_Challenge_TrainDataset.zip",
            "test": "https://iciar2018-challenge.grand-challenge.org/download/ICIAR2018_BACH_Challenge_TestDataset.zip"
        }
        
        os.makedirs("datasets/bach", exist_ok=True)
        
        for dataset_type, url in bach_urls.items():
            filename = f"datasets/bach/bach_{dataset_type}.zip"
            if download_file(url, filename):
                logger.info(f"Extracting {filename}...")
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(f"datasets/bach/{dataset_type}")
                os.remove(filename)  # Clean up zip file
        
        print("\n✅ BACH dataset downloaded and extracted")
        return True
        
    except Exception as e:
        logger.error(f"BACH dataset download failed: {str(e)}")
        print("\n⚠️  BACH Dataset Manual Download:")
        print("Visit: https://iciar2018-challenge.grand-challenge.org/dataset/")
        print("Register and download the dataset manually")
        return False

def download_sample_dataset():
    """Download a small sample dataset for testing"""
    try:
        # Create sample data structure
        os.makedirs("data/breast_cancer/benign", exist_ok=True)
        os.makedirs("data/breast_cancer/malignant", exist_ok=True)
        
        # Note: You would need to find actual sample images
        # This is just showing the structure
        print("\n📝 Sample Dataset Structure Created:")
        print("   data/breast_cancer/")
        print("   ├── benign/")
        print("   └── malignant/")
        print("\nPlace your breast cancer histology images in these folders")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample structure: {str(e)}")
        return False

def check_kaggle_datasets():
    """Check for breast cancer datasets on Kaggle"""
    print("\n🏆 Kaggle Breast Cancer Datasets:")
    print("=" * 40)
    print("1. 'Breast Histopathology Images'")
    print("   Command: kaggle datasets download -d paultimothymooney/breast-histopathology-images")
    print("   Size: ~1.2 GB, 277,524 images")
    print()
    print("2. 'BreakHis 400X'")
    print("   Command: kaggle datasets download -d fordaus/breakhis-400x")
    print("   Size: Subset of BreakHis at 400X magnification")
    print()
    print("3. 'Invasive Ductal Carcinoma (IDC) Detection'")
    print("   Command: kaggle datasets download -d paultimothymooney/breast-histopathology-images")
    print()
    print("📋 To use Kaggle:")
    print("   1. Install: pip install kaggle")
    print("   2. Setup API credentials: ~/.kaggle/kaggle.json")
    print("   3. Run the kaggle commands above")

def main():
    parser = argparse.ArgumentParser(description="Download breast cancer datasets")
    parser.add_argument("--dataset", choices=["breakhis", "bach", "sample", "kaggle"], 
                       help="Dataset to download/setup")
    parser.add_argument("--all", action="store_true", help="Show all dataset options")
    
    args = parser.parse_args()
    
    print("🎗️  Breast Cancer Dataset Manager")
    print("=" * 35)
    
    if args.all or not args.dataset:
        setup_breakhis_dataset()
        print()
        check_kaggle_datasets()
        print()
        download_sample_dataset()
    elif args.dataset == "breakhis":
        setup_breakhis_dataset()
    elif args.dataset == "bach":
        download_bach_dataset()
    elif args.dataset == "sample":
        download_sample_dataset()
    elif args.dataset == "kaggle":
        check_kaggle_datasets()

if __name__ == "__main__":
    main()