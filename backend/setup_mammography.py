#!/usr/bin/env python3
"""
Mammography Dataset Setup Script
This script helps you download and set up mammography datasets for breast cancer detection.
"""

import os
import sys
import argparse
from pathlib import Path
import requests
import shutil
from typing import Optional, List

def download_file(url: str, dest_path: str, chunk_size: int = 8192) -> bool:
    """Download a file with progress indication"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownloading... {progress:.1f}%", end='', flush=True)
        
        print()  # New line after download
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def setup_mias_dataset(download_dir: str = "./data/MIAS") -> bool:
    """Download and setup MIAS dataset (smallest, good for testing)"""
    print("📥 Setting up MIAS dataset...")
    
    os.makedirs(download_dir, exist_ok=True)
    
    # MIAS dataset URLs
    base_url = "http://peipa.essex.ac.uk/pix/mias/"
    files_to_download = [
        "all-mias.tar.gz",  # All images
        "info.txt"          # Information file
    ]
    
    for filename in files_to_download:
        url = base_url + filename
        dest_path = os.path.join(download_dir, filename)
        
        if os.path.exists(dest_path):
            print(f"✅ {filename} already exists")
            continue
        
        print(f"📥 Downloading {filename}...")
        if not download_file(url, dest_path):
            print(f"❌ Failed to download {filename}")
            return False
    
    # Extract tar.gz if it exists
    tar_path = os.path.join(download_dir, "all-mias.tar.gz")
    if os.path.exists(tar_path):
        print("📂 Extracting MIAS images...")
        import tarfile
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(download_dir)
        print("✅ MIAS dataset extracted successfully")
    
    return True

def setup_cbis_ddsm_info():
    """Provide information for CBIS-DDSM setup"""
    print("🏥 CBIS-DDSM Dataset Setup Information")
    print("=" * 50)
    print("CBIS-DDSM is the recommended mammography dataset but requires manual setup:")
    print()
    print("1️⃣  Create free account at The Cancer Imaging Archive (TCIA):")
    print("   https://www.cancerimagingarchive.net/")
    print()
    print("2️⃣  Download CBIS-DDSM dataset:")
    print("   https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM")
    print()
    print("3️⃣  Use TCIA's downloadable client or web interface")
    print()
    print("4️⃣  Extract to one of these locations:")
    print("   • ./data/CBIS-DDSM/")
    print("   • ~/Downloads/CBIS-DDSM/")
    print("   • ~/Documents/CBIS-DDSM/")
    print()
    print("5️⃣  Dataset structure should look like:")
    print("   CBIS-DDSM/")
    print("   ├── Mass/")
    print("   │   ├── Train/")
    print("   │   └── Test/")
    print("   └── Calc/")
    print("       ├── Train/")
    print("       └── Test/")
    print()

def setup_inbreast_info():
    """Provide information for INbreast setup"""
    print("🏥 INbreast Dataset Setup Information")
    print("=" * 50)
    print("INbreast is a high-quality mammography dataset:")
    print()
    print("1️⃣  Request access at:")
    print("   http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database")
    print()
    print("2️⃣  Fill out the research application form")
    print()
    print("3️⃣  Download the dataset after approval")
    print()
    print("4️⃣  Extract to one of these locations:")
    print("   • ./data/INbreast/")
    print("   • ~/Downloads/INbreast/")
    print("   • ~/Documents/INbreast/")
    print()
    print("5️⃣  Dataset contains DICOM files (.dcm)")
    print()

def check_dataset_status():
    """Check which datasets are currently available"""
    print("📊 Mammography Dataset Status")
    print("=" * 40)
    
    from dataset_utils import find_available_datasets, get_dataset_info
    
    available = find_available_datasets()
    available_dict = dict(available)
    
    datasets_to_check = ["cbis_ddsm", "ddsm", "inbreast", "mias", "synthetic"]
    
    for dataset_type in datasets_to_check:
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
        print()

def create_synthetic_mammograms(output_dir: str = "./data/mammography", count_per_class: int = 100):
    """Create synthetic mammography images for testing"""
    print(f"🎨 Creating {count_per_class * 2} synthetic mammography images...")
    
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    
    os.makedirs(os.path.join(output_dir, "benign"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "malignant"), exist_ok=True)
    
    for class_name in ["benign", "malignant"]:
        class_dir = os.path.join(output_dir, class_name)
        
        for i in range(count_per_class):
            # Create synthetic mammography-like image
            img_size = (512, 512)
            
            # Create base mammography texture
            img = Image.new('L', img_size, color=200)  # Light gray background
            draw = ImageDraw.Draw(img)
            
            # Add breast tissue texture
            for _ in range(100):
                x = np.random.randint(0, img_size[0])
                y = np.random.randint(0, img_size[1])
                size = np.random.randint(5, 20)
                color = np.random.randint(150, 250)
                draw.ellipse([x, y, x+size, y+size], fill=color)
            
            # Add class-specific features
            if class_name == "malignant":
                # Add irregular masses or microcalcifications
                for _ in range(np.random.randint(1, 4)):
                    x = np.random.randint(50, img_size[0]-50)
                    y = np.random.randint(50, img_size[1]-50)
                    size = np.random.randint(10, 30)
                    # Irregular shape for malignant
                    points = []
                    for angle in range(0, 360, 30):
                        r = size + np.random.randint(-5, 5)
                        px = x + r * np.cos(np.radians(angle))
                        py = y + r * np.sin(np.radians(angle))
                        points.append((px, py))
                    draw.polygon(points, fill=80)
            else:
                # Add smooth, regular masses for benign
                for _ in range(np.random.randint(0, 2)):
                    x = np.random.randint(50, img_size[0]-50)
                    y = np.random.randint(50, img_size[1]-50)
                    size = np.random.randint(15, 25)
                    draw.ellipse([x-size, y-size, x+size, y+size], fill=120)
            
            # Apply blur to simulate X-ray appearance
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            
            # Save image
            filename = f"synthetic_mammo_{class_name}_{i:03d}.png"
            img.save(os.path.join(class_dir, filename))
        
        print(f"✅ Created {count_per_class} synthetic {class_name} mammograms")

def main():
    parser = argparse.ArgumentParser(description="Setup mammography datasets for breast cancer detection")
    parser.add_argument("--setup-mias", action="store_true", help="Download and setup MIAS dataset")
    parser.add_argument("--info-cbis", action="store_true", help="Show CBIS-DDSM setup information")
    parser.add_argument("--info-inbreast", action="store_true", help="Show INbreast setup information")
    parser.add_argument("--check-status", action="store_true", help="Check dataset availability status")
    parser.add_argument("--create-synthetic", action="store_true", help="Create synthetic mammography data")
    parser.add_argument("--synthetic-count", type=int, default=100, help="Number of synthetic images per class")
    
    args = parser.parse_args()
    
    print("🏥 Mammography Dataset Setup for Breast Cancer Detection")
    print("=" * 60)
    
    if args.setup_mias:
        setup_mias_dataset()
        return 0
    
    if args.info_cbis:
        setup_cbis_ddsm_info()
        return 0
    
    if args.info_inbreast:
        setup_inbreast_info()
        return 0
    
    if args.create_synthetic:
        create_synthetic_mammograms(count_per_class=args.synthetic_count)
        return 0
    
    if args.check_status:
        check_dataset_status()
        return 0
    
    # Default: show general information
    print("🔬 Available Mammography Datasets for Breast Cancer Detection:")
    print()
    
    datasets_info = [
        ("CBIS-DDSM", "Largest, highest quality, DICOM format", "--info-cbis"),
        ("INbreast", "High quality, full-field digital", "--info-inbreast"), 
        ("MIAS", "Classic, smaller, good for testing", "--setup-mias"),
        ("Synthetic", "For demo/testing purposes", "--create-synthetic")
    ]
    
    for name, description, command in datasets_info:
        print(f"📊 {name}: {description}")
        print(f"   Use: python3 setup_mammography.py {command}")
        print()
    
    print("📋 To check what's currently available:")
    print("   python3 setup_mammography.py --check-status")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())