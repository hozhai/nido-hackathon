#!/usr/bin/env python3
"""
MIAS Dataset Organizer
This script reads the MIAS Info.txt file and separates the mammography images 
into benign and malignant categories based on the severity classification.
"""

import os
import shutil
from pathlib import Path

def parse_mias_info(info_file_path):
    """Parse the MIAS Info.txt file and return image classifications"""
    classifications = {}
    
    try:
        with open(info_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Info.txt file not found at {info_file_path}")
        return None
    
    # Find the start of the data (after the header)
    data_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('mdb') and len(line.split()) >= 3:
            data_start = i
            break
    
    print(f"Found data starting at line {data_start + 1}")
    
    for line_num, line in enumerate(lines[data_start:], start=data_start + 1):
        line = line.strip()
        if not line or not line.startswith('mdb'):
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
        
        image_id = parts[0]  # e.g., "mdb001"
        background = parts[1]  # F, G, or D
        abnormality_class = parts[2]  # CALC, CIRC, SPIC, MISC, ARCH, ASYM, NORM
        
        # Determine classification
        if abnormality_class == "NORM":
            # Normal images are benign
            classification = "benign"
        elif len(parts) >= 4:
            severity = parts[3]  # B (Benign) or M (Malignant)
            if severity == "B":
                classification = "benign"
            elif severity == "M":
                classification = "malignant"
            else:
                # If no severity specified but has abnormality, treat as benign
                classification = "benign"
        else:
            # Default to benign if no severity specified
            classification = "benign"
        
        classifications[image_id] = {
            'classification': classification,
            'background': background,
            'abnormality': abnormality_class,
            'severity': parts[3] if len(parts) >= 4 else 'NORMAL',
            'line_info': line
        }
        
        print(f"Line {line_num}: {image_id} -> {classification} (Class: {abnormality_class}, Severity: {parts[3] if len(parts) >= 4 else 'NORMAL'})")
    
    return classifications

def organize_mias_dataset(source_dir, output_dir, info_file_path):
    """Organize MIAS dataset into benign/malignant folders"""
    
    print(f"🔍 Parsing MIAS Info.txt file...")
    classifications = parse_mias_info(info_file_path)
    
    if not classifications:
        print("❌ Failed to parse Info.txt file")
        return False
    
    print(f"✅ Parsed {len(classifications)} image classifications")
    
    # Create output directories
    benign_dir = os.path.join(output_dir, "benign")
    malignant_dir = os.path.join(output_dir, "malignant")
    
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)
    
    benign_count = 0
    malignant_count = 0
    missing_images = []
    
    # Process each classification
    for image_id, info in classifications.items():
        source_image = os.path.join(source_dir, f"{image_id}.pgm")
        
        if not os.path.exists(source_image):
            missing_images.append(image_id)
            print(f"⚠️  Image not found: {source_image}")
            continue
        
        # Determine destination directory
        if info['classification'] == "benign":
            dest_dir = benign_dir
            benign_count += 1
        else:  # malignant
            dest_dir = malignant_dir
            malignant_count += 1
        
        # Copy image to destination
        dest_image = os.path.join(dest_dir, f"mias_{image_id}.pgm")
        
        try:
            shutil.copy2(source_image, dest_image)
            print(f"📋 {image_id} -> {info['classification']} ({info['abnormality']} - {info['severity']})")
        except Exception as e:
            print(f"❌ Failed to copy {image_id}: {e}")
    
    # Summary
    print(f"\n📊 MIAS Dataset Organization Complete!")
    print(f"✅ Benign images: {benign_count}")
    print(f"⚠️  Malignant images: {malignant_count}")
    if missing_images:
        print(f"❌ Missing images: {len(missing_images)} - {missing_images[:5]}")
    
    print(f"\n📁 Output directories:")
    print(f"   Benign: {benign_dir}")
    print(f"   Malignant: {malignant_dir}")
    
    return True

def main():
    # Configuration
    source_dir = "/home/zhai/Downloads"  # Where MIAS .pgm files are located
    output_dir = "/home/zhai/Documents/nido-hackathon/backend/data/mammography"  # Where to organize them
    info_file = "/home/zhai/Downloads/Info.txt"  # MIAS Info.txt file
    
    print("🏥 MIAS Dataset Organizer")
    print("=" * 50)
    print(f"📂 Source directory: {source_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📄 Info file: {info_file}")
    print()
    
    # Check if source files exist
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return 1
    
    if not os.path.exists(info_file):
        print(f"❌ Info.txt file not found: {info_file}")
        return 1
    
    # Count available .pgm files
    pgm_files = [f for f in os.listdir(source_dir) if f.endswith('.pgm')]
    print(f"🔍 Found {len(pgm_files)} .pgm files in source directory")
    
    if len(pgm_files) == 0:
        print("❌ No .pgm files found in source directory")
        return 1
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize the dataset
    success = organize_mias_dataset(source_dir, output_dir, info_file)
    
    if success:
        print("\n🎉 MIAS dataset successfully organized!")
        print("🚀 You can now train your mammography model with real MIAS data!")
        
        # Remove old synthetic model to force retraining
        old_model_path = "/home/zhai/Documents/nido-hackathon/backend/models/breast_cancer_mammography_classifier.pth"
        if os.path.exists(old_model_path):
            os.remove(old_model_path)
            print(f"🗑️  Removed old model to force retraining with MIAS data")
        
        print("\n📋 Next steps:")
        print("   1. Restart your backend server")
        print("   2. The model will auto-train with real MIAS mammography data")
        print("   3. Check model status via API: curl http://localhost:8000/model/status")
        
        return 0
    else:
        print("❌ Failed to organize MIAS dataset")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())