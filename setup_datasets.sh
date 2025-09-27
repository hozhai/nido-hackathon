#!/bin/bash

# Cancer Tissue Dataset Setup Script
# This script helps set up cancer tissue datasets for the application

echo "🏥 Cancer Tissue Detection - Dataset Setup"
echo "========================================="

# Create data directory structure
echo "📁 Creating data directory structure..."
mkdir -p backend/data/cancer_samples/{benign,malignant}
mkdir -p backend/data/downloads

echo "✅ Directory structure created successfully!"
echo ""

echo "📊 Available Cancer Tissue Datasets:"
echo ""

echo "1. BreakHis Dataset (Breast Cancer Histopathological Images)"
echo "   - 7,909 microscopic images"
echo "   - 2,480 benign + 5,429 malignant samples"
echo "   - Multiple magnifications (40X, 100X, 200X, 400X)"
echo "   - Download: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/"
echo ""

echo "2. PatchCamelyon (PCam) Dataset"
echo "   - 327,680 histopathologic images (96x96px)"
echo "   - Lymph node sections with metastatic tissue detection"
echo "   - GitHub: https://github.com/basveeling/pcam"
echo ""

echo "3. Camelyon16/17 Challenge Dataset"
echo "   - Whole-slide images of histopathologic scans"
echo "   - Metastatic cancer detection in sentinel lymph nodes"
echo "   - Website: https://camelyon17.grand-challenge.org/"
echo ""

echo "4. TCGA (The Cancer Genome Atlas)"
echo "   - Large-scale cancer genomics with tissue slide images"
echo "   - Multiple cancer types available"
echo "   - Portal: https://portal.gdc.cancer.gov/"
echo ""

echo "🛠️  Setup Instructions:"
echo ""
echo "For BreakHis Dataset:"
echo "1. Visit the BreakHis website and request dataset access"
echo "2. Download BreaKHis_v1.tar.gz"
echo "3. Extract to backend/data/downloads/"
echo "4. Run the organize_breakhis.py script (see below)"
echo ""

echo "For PCam Dataset:"
echo "1. Clone: git clone https://github.com/basveeling/pcam.git"
echo "2. Follow the download instructions in the repository"
echo "3. Use the provided scripts to convert H5 files to images"
echo ""

# Create Python script to organize BreakHis dataset
echo "📝 Creating dataset organization script..."

cat > backend/organize_breakhis.py << 'EOF'
#!/usr/bin/env python3
"""
Script to organize the BreakHis dataset into training structure
"""

import os
import shutil
import sys
from pathlib import Path

def organize_breakhis_dataset(source_dir, target_dir):
    """
    Organize BreakHis dataset into benign/malignant structure
    
    Args:
        source_dir: Path to extracted BreakHis dataset
        target_dir: Path to organized dataset directory
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directories
    benign_dir = target_path / "benign"
    malignant_dir = target_path / "malignant"
    benign_dir.mkdir(parents=True, exist_ok=True)
    malignant_dir.mkdir(parents=True, exist_ok=True)
    
    # Walk through source directory
    image_count = {"benign": 0, "malignant": 0}
    
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = Path(root) / file
                
                # Determine if image is benign or malignant based on path
                if 'benign' in root.lower() or 'B' in file:
                    dest_dir = benign_dir
                    category = "benign"
                elif 'malignant' in root.lower() or 'M' in file:
                    dest_dir = malignant_dir  
                    category = "malignant"
                else:
                    print(f"Unknown category for {file_path}")
                    continue
                
                # Copy file to target directory
                dest_file = dest_dir / f"{category}_{image_count[category]:05d}_{file}"
                shutil.copy2(file_path, dest_file)
                image_count[category] += 1
                
                if image_count[category] % 100 == 0:
                    print(f"Processed {image_count[category]} {category} images...")
    
    print(f"\nDataset organization complete!")
    print(f"Benign images: {image_count['benign']}")
    print(f"Malignant images: {image_count['malignant']}")
    print(f"Total images: {sum(image_count.values())}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python organize_breakhis.py <source_dir> <target_dir>")
        print("Example: python organize_breakhis.py ./downloads/BreaKHis_v1 ./data/cancer_samples")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        sys.exit(1)
    
    organize_breakhis_dataset(source_dir, target_dir)
EOF

chmod +x backend/organize_breakhis.py

echo "✅ Dataset organization script created: backend/organize_breakhis.py"
echo ""

# Create sample dataset downloader script
echo "📝 Creating sample dataset downloader..."

cat > backend/download_samples.py << 'EOF'
#!/usr/bin/env python3
"""
Download sample cancer tissue images for demonstration
"""

import os
import requests
from pathlib import Path
import time

def download_file(url, filepath):
    """Download a file from URL to filepath"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    
    # Note: These are placeholder URLs - replace with actual cancer tissue image URLs
    # or use local sample images
    
    sample_images = {
        'benign': [
            # Add URLs to benign tissue sample images
            # Example: 'https://example.com/sample_benign_1.jpg',
        ],
        'malignant': [
            # Add URLs to malignant tissue sample images  
            # Example: 'https://example.com/sample_malignant_1.jpg',
        ]
    }
    
    base_dir = Path("data/cancer_samples")
    
    for category, urls in sample_images.items():
        category_dir = base_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {category} samples...")
        
        for i, url in enumerate(urls):
            filename = f"sample_{category}_{i+1}.jpg"
            filepath = category_dir / filename
            
            if download_file(url, filepath):
                print(f"  ✅ Downloaded {filename}")
            else:
                print(f"  ❌ Failed to download {filename}")
            
            time.sleep(0.5)  # Be polite to servers
    
    print("\n📋 Sample Dataset Setup Complete!")
    print("Note: This creates a minimal sample dataset.")
    print("For production use, download full datasets from official sources.")

if __name__ == "__main__":
    create_sample_dataset()
EOF

chmod +x backend/download_samples.py

echo "✅ Sample downloader script created: backend/download_samples.py"
echo ""

echo "🚀 Next Steps:"
echo ""
echo "1. Install Python dependencies:"
echo "   cd backend && pip install -r requirements.txt"
echo ""
echo "2. Start the backend server:"
echo "   cd backend && python main.py"
echo ""
echo "3. Start the frontend (in another terminal):"
echo "   cd frontend && npm run dev"
echo ""
echo "4. For real datasets:"
echo "   - Download BreakHis dataset and run organize_breakhis.py"
echo "   - Or use the web interface to get dataset information"
echo ""
echo "5. The pre-trained model is ready to use immediately!"
echo "   - Upload histopathological images through the web interface"
echo "   - Get benign/malignant predictions with confidence scores"
echo ""

echo "✨ Setup complete! Your cancer tissue detection system is ready."