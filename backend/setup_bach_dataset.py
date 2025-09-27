#!/usr/bin/env python3
"""
BACH Dataset Setup and Validation Script
This script helps you set up the BACH dataset for breast cancer detection training.
"""

import os
import sys
from pathlib import Path
import argparse
from PIL import Image
import numpy as np

def find_bach_dataset(search_paths=None):
    """Find BACH dataset in common locations"""
    if search_paths is None:
        search_paths = [
            "./BACH",
            "./bach_dataset", 
            "../BACH",
            "../bach_dataset",
            "../../BACH",
            "../../bach_dataset",
            os.path.expanduser("~/Downloads/BACH"),
            os.path.expanduser("~/Downloads/bach_dataset"),
            os.path.expanduser("~/Documents/BACH"),
            os.path.expanduser("~/Documents/bach_dataset"),
            "/tmp/BACH",
            "/tmp/bach_dataset"
        ]
    
    print("🔍 Searching for BACH dataset...")
    for path in search_paths:
        if os.path.exists(path):
            # Check for expected structure
            photos_dir = os.path.join(path, "Photos")
            if os.path.exists(photos_dir):
                print(f"✅ Found BACH dataset at: {path}")
                return path
            
            # Check for alternative structure
            bach_classes = ["Normal", "Benign", "InSitu", "Invasive"]
            if any(os.path.exists(os.path.join(path, cls)) for cls in bach_classes):
                print(f"✅ Found BACH dataset (alternative structure) at: {path}")
                return path
    
    print("❌ BACH dataset not found in common locations")
    return None

def validate_bach_structure(bach_path):
    """Validate BACH dataset structure and count images"""
    print(f"\n📊 Validating BACH dataset structure at: {bach_path}")
    
    bach_classes = ["Normal", "Benign", "InSitu", "Invasive"]
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    
    total_images = 0
    class_counts = {}
    
    # Check standard structure first
    photos_dir = os.path.join(bach_path, "Photos")
    if os.path.exists(photos_dir):
        base_dir = photos_dir
        print("📁 Using standard BACH structure: BACH/Photos/[Class]/")
    else:
        base_dir = bach_path
        print("📁 Using alternative BACH structure: BACH/[Class]/")
    
    for bach_class in bach_classes:
        class_dir = os.path.join(base_dir, bach_class)
        if os.path.exists(class_dir):
            # Count images in this class
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(image_extensions)]
            class_count = len(image_files)
            class_counts[bach_class] = class_count
            total_images += class_count
            
            print(f"  📸 {bach_class}: {class_count} images")
            
            # Check a sample image
            if image_files:
                sample_image_path = os.path.join(class_dir, image_files[0])
                try:
                    with Image.open(sample_image_path) as img:
                        print(f"    📏 Sample image size: {img.size}")
                except Exception as e:
                    print(f"    ⚠️  Error reading sample image: {e}")
        else:
            print(f"  ❌ {bach_class} directory not found")
            class_counts[bach_class] = 0
    
    print(f"\n📈 Total images found: {total_images}")
    
    # Map to binary classification
    benign_count = class_counts.get("Normal", 0) + class_counts.get("Benign", 0)
    malignant_count = class_counts.get("InSitu", 0) + class_counts.get("Invasive", 0)
    
    print(f"🔬 For binary classification:")
    print(f"  • Benign (Normal + Benign): {benign_count} images")
    print(f"  • Malignant (InSitu + Invasive): {malignant_count} images")
    
    if total_images == 0:
        print("❌ No images found! Please check your BACH dataset structure.")
        return False
    
    if benign_count == 0 or malignant_count == 0:
        print("⚠️  Unbalanced dataset: one class has no images!")
    
    return True

def setup_data_directory(bach_path, data_dir="./data"):
    """Set up data directory structure for the model"""
    print(f"\n🔧 Setting up data directory structure in: {data_dir}")
    
    # Create breast_cancer directory structure
    breast_cancer_dir = os.path.join(data_dir, "breast_cancer")
    os.makedirs(breast_cancer_dir, exist_ok=True)
    
    # Create symbolic links or copy instructions
    bach_abs_path = os.path.abspath(bach_path)
    
    print(f"📁 Created directory: {breast_cancer_dir}")
    print(f"🔗 To use BACH dataset, ensure it's accessible from: {bach_abs_path}")
    
    # Create a marker file indicating BACH dataset location
    bach_info_file = os.path.join(breast_cancer_dir, "bach_dataset_path.txt")
    with open(bach_info_file, "w") as f:
        f.write(f"BACH dataset location: {bach_abs_path}\n")
        f.write(f"Setup date: {os.popen('date').read().strip()}\n")
    
    print(f"💾 Created info file: {bach_info_file}")

def test_model_integration():
    """Test if the breast cancer model can find and use BACH dataset"""
    print("\n🧪 Testing model integration...")
    
    try:
        from breast_cancer_model import BreastCancerClassifier
        
        # Create a test instance (but don't train)
        print("📦 Importing BreastCancerClassifier...")
        
        # Check if the model would find BACH dataset
        data_dir = "./data"
        breast_cancer_dir = os.path.join(data_dir, "breast_cancer")
        
        # Simulate the search that the model does
        bach_possible_paths = [
            os.path.join(data_dir, "BACH"),
            os.path.join(data_dir, "bach_dataset"),
            os.path.join(".", "BACH"),
            os.path.join(".", "bach_dataset"),
        ]
        
        found_path = None
        for path in bach_possible_paths:
            if os.path.exists(path):
                photos_dir = os.path.join(path, "Photos")
                if os.path.exists(photos_dir):
                    found_path = path
                    break
        
        if found_path:
            print(f"✅ Model will find BACH dataset at: {found_path}")
        else:
            print("⚠️  Model might not find BACH dataset automatically.")
            print("   Consider creating a symlink:")
            print(f"   ln -s /path/to/your/BACH {os.path.join(data_dir, 'BACH')}")
        
    except ImportError as e:
        print(f"❌ Could not import BreastCancerClassifier: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup and validate BACH dataset for breast cancer detection")
    parser.add_argument("--bach-path", type=str, help="Path to BACH dataset directory")
    parser.add_argument("--search-only", action="store_true", help="Only search for BACH dataset")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing dataset")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory for model")
    
    args = parser.parse_args()
    
    print("🎗️ BACH Dataset Setup for Breast Cancer Detection")
    print("=" * 50)
    
    # Find BACH dataset
    if args.bach_path and os.path.exists(args.bach_path):
        bach_path = args.bach_path
        print(f"✅ Using provided BACH dataset path: {bach_path}")
    else:
        bach_path = find_bach_dataset()
    
    if not bach_path:
        print("\n❓ BACH Dataset Not Found!")
        print("Please download the BACH dataset from:")
        print("   https://iciar2018-challenge.grand-challenge.org/dataset/")
        print("\nExtract it to one of these locations:")
        print("   • ./BACH")
        print("   • ./bach_dataset") 
        print("   • ~/Downloads/BACH")
        print("   • Or specify with --bach-path /your/path/to/BACH")
        return 1
    
    if args.search_only:
        return 0
    
    # Validate dataset structure
    if not validate_bach_structure(bach_path):
        print("\n❌ BACH dataset validation failed!")
        return 1
    
    if args.validate_only:
        return 0
    
    # Setup data directory
    setup_data_directory(bach_path, args.data_dir)
    
    # Test model integration
    test_model_integration()
    
    print("\n✅ BACH dataset setup complete!")
    print("🚀 You can now train the breast cancer model with real BACH data.")
    print("\nNext steps:")
    print("   1. Remove existing synthetic model: rm models/breast_cancer_classifier.pth")
    print("   2. Restart the backend server to trigger BACH dataset training")
    print("   3. Monitor the training logs for BACH dataset usage")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())