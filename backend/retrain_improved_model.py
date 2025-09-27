#!/usr/bin/env python3
"""
Retrain the mammography model with accuracy improvements
"""

import os
import sys
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_mammography_model import ImprovedMammographyClassifier, AdvancedMammographyTransforms
from breast_cancer_model import BreastCancerDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_mammography_data(data_dir):
    """Load the MIAS mammography data"""
    mammography_dir = os.path.join(data_dir, "mammography")
    benign_dir = os.path.join(mammography_dir, "benign")
    malignant_dir = os.path.join(mammography_dir, "malignant")
    
    image_paths = []
    labels = []
    
    # Load benign images
    if os.path.exists(benign_dir):
        for img_file in os.listdir(benign_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm', '.dcm')):
                image_paths.append(os.path.join(benign_dir, img_file))
                labels.append(0)  # 0 for benign
    
    # Load malignant images
    if os.path.exists(malignant_dir):
        for img_file in os.listdir(malignant_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm', '.dcm')):
                image_paths.append(os.path.join(malignant_dir, img_file))
                labels.append(1)  # 1 for malignant
    
    logger.info(f"Loaded {len(image_paths)} mammography images ({labels.count(0)} benign, {labels.count(1)} malignant)")
    
    return image_paths, labels

def create_balanced_dataloaders(image_paths, labels, batch_size=8, test_size=0.2):
    """Create balanced dataloaders with weighted sampling"""
    # Stratified split to maintain class distribution
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    logger.info(f"Training set: {len(train_paths)} images ({train_labels.count(0)} benign, {train_labels.count(1)} malignant)")
    logger.info(f"Validation set: {len(val_paths)} images ({val_labels.count(0)} benign, {val_labels.count(1)} malignant)")
    
    # Create datasets with advanced transforms
    train_transform = AdvancedMammographyTransforms.get_training_transforms()
    val_transform = AdvancedMammographyTransforms.get_validation_transforms()
    
    train_dataset = BreastCancerDataset(train_paths, train_labels, train_transform)
    val_dataset = BreastCancerDataset(val_paths, val_labels, val_transform)
    
    # Create weighted sampler to handle class imbalance
    class_counts = [train_labels.count(0), train_labels.count(1)]
    class_weights = [1.0 / count for count in class_counts]
    sample_weights = [class_weights[label] for label in train_labels]
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, 
                                   num_samples=len(sample_weights), 
                                   replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                             batch_size=batch_size, 
                             sampler=sampler,  # Use weighted sampling
                             drop_last=True)
    
    val_loader = DataLoader(val_dataset, 
                           batch_size=batch_size, 
                           shuffle=False, 
                           drop_last=True)
    
    return train_loader, val_loader

def main():
    logger.info("🔄 Starting Enhanced Mammography Model Training")
    logger.info("=" * 60)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    data_dir = os.path.join(script_dir, "data")
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    logger.info("📊 Loading MIAS mammography dataset...")
    image_paths, labels = load_mammography_data(data_dir)
    
    if len(image_paths) == 0:
        logger.error("❌ No mammography images found! Please ensure MIAS data is organized.")
        return False
    
    # Check class distribution
    benign_count = labels.count(0)
    malignant_count = labels.count(1)
    imbalance_ratio = benign_count / malignant_count if malignant_count > 0 else float('inf')
    
    logger.info(f"📈 Dataset Analysis:")
    logger.info(f"   Benign images: {benign_count}")
    logger.info(f"   Malignant images: {malignant_count}")
    logger.info(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3.0:
        logger.warning(f"⚠️  High class imbalance detected! Applying advanced balancing techniques...")
    
    # Create balanced data loaders
    logger.info("🔄 Creating balanced data loaders...")
    train_loader, val_loader = create_balanced_dataloaders(image_paths, labels, batch_size=8)
    
    # Initialize improved model
    logger.info("🏗️  Initializing enhanced mammography model...")
    classifier = ImprovedMammographyClassifier(model_dir=model_dir, data_dir=data_dir)
    
    # Train with improvements
    logger.info("🚀 Starting enhanced training...")
    logger.info("💡 Using: Focal Loss, Weighted Sampling, Advanced Augmentation, LR Scheduling")
    
    try:
        best_acc, best_sensitivity = classifier.train_with_improvements(train_loader, val_loader)
        
        logger.info("=" * 60)
        logger.info("✅ Enhanced Training Completed Successfully!")
        logger.info(f"🎯 Best Validation Accuracy: {best_acc:.2f}%")
        logger.info(f"🔍 Best Sensitivity (Malignant Detection): {best_sensitivity:.2f}%")
        logger.info("📁 Enhanced model saved as 'enhanced_mammography_classifier.pth'")
        logger.info("=" * 60)
        
        # Test prediction on a sample image
        logger.info("🧪 Testing enhanced model...")
        test_image_path = None
        
        # Find a sample malignant image for testing
        malignant_dir = os.path.join(data_dir, "mammography", "malignant")
        if os.path.exists(malignant_dir):
            malignant_files = [f for f in os.listdir(malignant_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.pgm'))]
            if malignant_files:
                test_image_path = os.path.join(malignant_dir, malignant_files[0])
        
        if test_image_path:
            result = classifier.predict_with_confidence(test_image_path)
            logger.info(f"📋 Sample prediction on {os.path.basename(test_image_path)}:")
            logger.info(f"   Predicted: {result['predicted_class']} ({result['confidence']:.1%} confidence)")
            logger.info(f"   Risk Level: {result['risk_level']}")
            logger.info(f"   Malignant Probability: {result['probabilities']['malignant']:.1%}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Enhanced mammography model is ready!")
        print("💡 The model now has improved accuracy for malignant detection!")
    else:
        print("\n❌ Training failed. Please check the logs above.")
        sys.exit(1)