#!/usr/bin/env python3
"""
Test Enhanced Mammography Model vs Standard Model
Compare accuracy improvements, especially for malignant detection
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add current directory to path  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from improved_mammography_model import ImprovedMammographyClassifier
    from breast_cancer_model import BreastCancerClassifier
except ImportError as e:
    print(f"Error importing models: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_predictions(model, model_name, test_images):
    """Test model predictions on sample images"""
    results = []
    
    print(f"\n🧪 Testing {model_name} Model:")
    print("=" * 50)
    
    for image_path, true_label in test_images:
        malignant_prob = 0  # Initialize default value
        try:
            if hasattr(model, 'predict_with_confidence'):
                # Enhanced model
                result = model.predict_with_confidence(image_path)
                predicted_class = result["predicted_class"]
                confidence = result["confidence"] * 100
                malignant_prob = result["probabilities"]["malignant"] * 100
                risk_level = result.get("risk_level", "N/A")
                
                print(f"📋 {Path(image_path).name}")
                print(f"   True Label: {true_label}")
                print(f"   Predicted: {predicted_class} ({confidence:.1f}% confidence)")
                print(f"   Malignant Probability: {malignant_prob:.1f}%")
                print(f"   Risk Level: {risk_level}")
                
            else:
                # Standard model
                result = model.predict(image_path)
                predicted_class = result["predicted_class"]
                confidence = result["confidence"]
                malignant_prob = result.get("probabilities", {}).get("malignant", 0)
                
                print(f"📋 {Path(image_path).name}")
                print(f"   True Label: {true_label}")
                print(f"   Predicted: {predicted_class} ({confidence:.1f}% confidence)")
            
            # Record result
            correct = (predicted_class == true_label)
            results.append({
                'image': Path(image_path).name,
                'true_label': true_label,
                'predicted': predicted_class,
                'confidence': confidence if not hasattr(model, 'predict_with_confidence') else result["confidence"] * 100,
                'correct': correct,
                'malignant_prob': malignant_prob
            })
            
            status = "✅ CORRECT" if correct else "❌ INCORRECT"
            print(f"   Result: {status}")
            print()
            
        except Exception as e:
            print(f"❌ Error testing {Path(image_path).name}: {e}")
            results.append({
                'image': Path(image_path).name,
                'true_label': true_label,
                'predicted': 'ERROR',
                'confidence': 0,
                'correct': False,
                'malignant_prob': 0
            })
    
    return results

def analyze_results(results, model_name):
    """Analyze prediction results"""
    if not results:
        return
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100
    
    # Analyze by class
    benign_results = [r for r in results if r['true_label'] == 'benign']
    malignant_results = [r for r in results if r['true_label'] == 'malignant']
    
    benign_correct = sum(1 for r in benign_results if r['correct'])
    malignant_correct = sum(1 for r in malignant_results if r['correct'])
    
    benign_accuracy = (benign_correct / len(benign_results) * 100) if benign_results else 0
    malignant_accuracy = (malignant_correct / len(malignant_results) * 100) if malignant_results else 0
    
    print(f"📊 {model_name} Results Summary:")
    print(f"   Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"   Benign Accuracy: {benign_accuracy:.1f}% ({benign_correct}/{len(benign_results)})")
    print(f"   Malignant Accuracy: {malignant_accuracy:.1f}% ({malignant_correct}/{len(malignant_results)})")
    
    # Average confidence for correct malignant predictions
    correct_malignant = [r for r in malignant_results if r['correct']]
    if correct_malignant:
        avg_malignant_confidence = sum(r['malignant_prob'] for r in correct_malignant) / len(correct_malignant)
        print(f"   Avg Malignant Detection Confidence: {avg_malignant_confidence:.1f}%")
    
    return {
        'accuracy': accuracy,
        'benign_accuracy': benign_accuracy,
        'malignant_accuracy': malignant_accuracy,
        'malignant_sensitivity': malignant_accuracy  # Same as malignant accuracy in this context
    }

def main():
    print("🔬 Enhanced vs Standard Mammography Model Comparison")
    print("=" * 60)
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    data_dir = os.path.join(script_dir, "data")
    mammography_dir = os.path.join(data_dir, "mammography")
    
    # Collect test images
    test_images = []
    
    # Get some benign images
    benign_dir = os.path.join(mammography_dir, "benign")
    if os.path.exists(benign_dir):
        benign_files = [f for f in os.listdir(benign_dir) if f.endswith(('.pgm', '.png', '.jpg'))][:3]
        for f in benign_files:
            test_images.append((os.path.join(benign_dir, f), 'benign'))
    
    # Get some malignant images
    malignant_dir = os.path.join(mammography_dir, "malignant")
    if os.path.exists(malignant_dir):
        malignant_files = [f for f in os.listdir(malignant_dir) if f.endswith(('.pgm', '.png', '.jpg'))][:3]
        for f in malignant_files:
            test_images.append((os.path.join(malignant_dir, f), 'malignant'))
    
    if not test_images:
        print("❌ No test images found! Please ensure MIAS data is organized.")
        return
    
    print(f"📁 Found {len(test_images)} test images")
    print(f"   Benign: {sum(1 for _, label in test_images if label == 'benign')}")
    print(f"   Malignant: {sum(1 for _, label in test_images if label == 'malignant')}")
    
    results = {}
    
    # Test Standard Model
    print("\n🔧 Loading Standard Model...")
    try:
        standard_model = BreastCancerClassifier(model_dir=model_dir, data_dir=data_dir)
        if standard_model.is_trained:
            results['standard'] = test_model_predictions(standard_model, "Standard ResNet18", test_images)
            standard_stats = analyze_results(results['standard'], "Standard Model")
        else:
            print("⚠️ Standard model not trained. Skipping.")
            standard_stats = None
    except Exception as e:
        print(f"❌ Error loading standard model: {e}")
        standard_stats = None
    
    # Test Enhanced Model
    enhanced_model_path = os.path.join(model_dir, "enhanced_mammography_classifier.pth")
    if os.path.exists(enhanced_model_path):
        print("\n🚀 Loading Enhanced Model...")
        try:
            enhanced_model = ImprovedMammographyClassifier(model_dir=model_dir, data_dir=data_dir)
            
            # Load saved model
            checkpoint = torch.load(enhanced_model_path, map_location=enhanced_model.device)
            enhanced_model.model.load_state_dict(checkpoint['model_state_dict'])
            enhanced_model.is_trained = True
            
            results['enhanced'] = test_model_predictions(enhanced_model, "Enhanced EfficientNet-B3", test_images)
            enhanced_stats = analyze_results(results['enhanced'], "Enhanced Model")
            
        except Exception as e:
            print(f"❌ Error loading enhanced model: {e}")
            enhanced_stats = None
    else:
        print("❌ Enhanced model not found. Please run retrain_improved_model.py first.")
        enhanced_stats = None
    
    # Compare results
    print("\n" + "=" * 60)
    print("🏆 COMPARISON RESULTS")
    print("=" * 60)
    
    if standard_stats and enhanced_stats:
        print(f"📈 Overall Accuracy:")
        print(f"   Standard Model: {standard_stats['accuracy']:.1f}%")
        print(f"   Enhanced Model: {enhanced_stats['accuracy']:.1f}%")
        print(f"   Improvement: {enhanced_stats['accuracy'] - standard_stats['accuracy']:+.1f}%")
        
        print(f"\n🔍 Malignant Detection (Sensitivity):")
        print(f"   Standard Model: {standard_stats['malignant_accuracy']:.1f}%")
        print(f"   Enhanced Model: {enhanced_stats['malignant_accuracy']:.1f}%")
        print(f"   Improvement: {enhanced_stats['malignant_accuracy'] - standard_stats['malignant_accuracy']:+.1f}%")
        
        if enhanced_stats['malignant_accuracy'] > standard_stats['malignant_accuracy']:
            print("\n✅ Enhanced model shows IMPROVED malignant detection!")
        elif enhanced_stats['malignant_accuracy'] == standard_stats['malignant_accuracy']:
            print("\n➖ Enhanced model shows EQUAL malignant detection")
        else:
            print("\n⚠️ Enhanced model needs further tuning")
    
    print("\n💡 Key Improvements in Enhanced Model:")
    print("   • EfficientNet-B3 architecture (vs ResNet18)")
    print("   • Focal Loss for class imbalance")
    print("   • Weighted sampling for balanced training")
    print("   • Advanced medical image augmentation")
    print("   • Test-time augmentation for predictions")
    print("   • Enhanced risk assessment and recommendations")
    
    print("\n🎯 For your specific case where malignant was predicted as benign:")
    print("   The enhanced model should now have better sensitivity for malignant cases")
    print("   due to the specialized training techniques applied.")

if __name__ == "__main__":
    main()