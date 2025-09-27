"""
Enhanced Breast Cancer Mammography Classifier
Implements multiple accuracy improvements for better malignant detection
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageEnhance, ImageFilter
import os
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    Focuses learning on hard-to-classify examples (like malignant cases)
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class EfficientMammographyNet(nn.Module):
    """
    Enhanced mammography classification network using EfficientNet backbone
    with custom head optimized for medical imaging
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientMammographyNet, self).__init__()
        
        # Use EfficientNet-B3 for better feature extraction
        try:
            from torchvision.models import efficientnet_b3
            self.backbone = efficientnet_b3(pretrained=pretrained)
            feature_size = 1536  # EfficientNet-B3 feature size
        except ImportError:
            # Fallback to ResNet50 if EfficientNet not available
            logger.warning("EfficientNet not available, using ResNet50")
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_size = 2048
        
        # Remove the original classifier
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
            
        # Custom mammography-specific head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights for the classifier
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return self.classifier(features)

class AdvancedMammographyTransforms:
    """Advanced preprocessing transforms optimized for mammography analysis"""
    
    @staticmethod
    def get_training_transforms():
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((256, 256)),  # Intermediate resize
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Better crop strategy
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15, fill=0),  # Slightly more rotation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
            
            # Medical imaging specific augmentations
            transforms.Lambda(lambda x: AdvancedMammographyTransforms.enhance_contrast(x)),
            transforms.Lambda(lambda x: AdvancedMammographyTransforms.add_noise(x)),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_validation_transforms():
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def enhance_contrast(image, prob=0.3):
        """Randomly enhance contrast for better abnormality visibility"""
        if random.random() < prob:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(1.1, 1.4)
            return enhancer.enhance(factor)
        return image
    
    @staticmethod
    def add_noise(image, prob=0.2):
        """Add subtle noise to improve robustness"""
        if random.random() < prob:
            np_image = np.array(image)
            noise = np.random.normal(0, 5, np_image.shape).astype(np.uint8)
            noisy_image = np.clip(np_image.astype(int) + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_image)
        return image

class ImprovedMammographyClassifier:
    """
    Enhanced mammography classifier with multiple accuracy improvements:
    1. Advanced architecture (EfficientNet/ResNet50)
    2. Focal Loss for class imbalance
    3. Weighted sampling
    4. Advanced data augmentation
    5. Learning rate scheduling
    6. Ensemble predictions (optional)
    """
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_trained = False
        self.classes = ["benign", "malignant"]
        
        # Training parameters optimized for imbalanced data
        self.learning_rate = 0.0001  # Lower LR for better convergence
        self.batch_size = 8  # Smaller batch for limited data
        self.epochs = 25  # More epochs for better learning
        self.early_stopping_patience = 7
        
        # Setup transforms
        self.train_transform = AdvancedMammographyTransforms.get_training_transforms()
        self.val_transform = AdvancedMammographyTransforms.get_validation_transforms()
        
        # Initialize model
        self._setup_model()
        
        logger.info(f"ImprovedMammographyClassifier initialized on {self.device}")
    
    def _setup_model(self):
        """Initialize the enhanced model architecture"""
        self.model = EfficientMammographyNet(num_classes=2, pretrained=True)
        self.model = self.model.to(self.device)
    
    def train_with_improvements(self, train_loader, val_loader):
        """
        Train the model with all accuracy improvements
        """
        logger.info("Starting improved mammography training...")
        
        # Calculate class weights for handling imbalance
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.numpy())
        
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(train_labels), 
                                           y=train_labels)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        logger.info(f"Class weights - Benign: {class_weights[0]:.3f}, Malignant: {class_weights[1]:.3f}")
        
        # Use Focal Loss for better handling of class imbalance
        criterion = FocalLoss(alpha=class_weights_tensor[1], gamma=2.0)
        
        # Advanced optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                    patience=3)
        
        best_val_acc = 0.0
        best_val_sensitivity = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            
            # Validation phase with detailed metrics
            val_acc, val_sensitivity, val_specificity, val_loss = self._detailed_validation(val_loader, criterion)
            
            logger.info(f'Epoch [{epoch+1}/{self.epochs}] - '
                       f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% - '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                       f'Sensitivity: {val_sensitivity:.2f}%, Specificity: {val_specificity:.2f}%')
            
            # Save best model based on balanced metric (focus on sensitivity for malignant detection)
            balanced_score = 0.6 * val_sensitivity + 0.4 * val_acc
            best_balanced_score = 0.6 * best_val_sensitivity + 0.4 * best_val_acc
            
            if balanced_score > best_balanced_score:
                best_val_acc = val_acc
                best_val_sensitivity = val_sensitivity
                self._save_enhanced_model()
                epochs_without_improvement = 0
                logger.info(f"New best model saved! Balanced score: {balanced_score:.2f}")
            else:
                epochs_without_improvement += 1
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Early stopping
            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.is_trained = True
        logger.info(f"Training completed! Best Val Acc: {best_val_acc:.2f}%, "
                   f"Best Sensitivity: {best_val_sensitivity:.2f}%")
        
        return best_val_acc, best_val_sensitivity
    
    def _detailed_validation(self, val_loader, criterion):
        """Validation with detailed metrics including sensitivity/specificity"""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate detailed metrics
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0  # True positive rate (malignant detection)
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0  # True negative rate (benign detection)
        
        val_loss_avg = val_loss / len(val_loader)
        
        return accuracy, sensitivity, specificity, val_loss_avg
    
    def predict_with_confidence(self, image_path: str) -> Dict:
        """
        Enhanced prediction with confidence analysis and multiple perspectives
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Test-time augmentation for more robust predictions
        predictions = []
        confidences = []
        
        # Original prediction
        input_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            predictions.append(probs.cpu().numpy()[0])
        
        # Horizontal flip prediction
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        input_tensor = self.val_transform(flipped_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            predictions.append(probs.cpu().numpy()[0])
        
        # Average predictions (ensemble)
        avg_probs = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_probs)
        confidence = float(avg_probs[predicted_class])
        
        # Enhanced confidence interpretation
        risk_level = self._interpret_risk(avg_probs, predicted_class)
        
        result = {
            "predicted_class": self.classes[predicted_class],
            "confidence": confidence,
            "probabilities": {
                "benign": float(avg_probs[0]),
                "malignant": float(avg_probs[1])
            },
            "risk_level": risk_level,
            "recommendation": self._get_medical_recommendation(avg_probs, predicted_class),
            "technical_details": {
                "model_type": "Enhanced EfficientNet/ResNet50",
                "test_time_augmentation": True,
                "ensemble_predictions": len(predictions)
            }
        }
        
        return result
    
    def _interpret_risk(self, probabilities, predicted_class):
        """Enhanced risk interpretation"""
        malignant_prob = probabilities[1]
        
        if malignant_prob >= 0.7:
            return "HIGH_RISK"
        elif malignant_prob >= 0.3:
            return "MODERATE_RISK"
        elif malignant_prob >= 0.1:
            return "LOW_RISK"
        else:
            return "MINIMAL_RISK"
    
    def _get_medical_recommendation(self, probabilities, predicted_class):
        """Provide medical recommendations based on risk assessment"""
        malignant_prob = probabilities[1]
        
        if malignant_prob >= 0.7:
            return "URGENT: High probability of malignancy. Immediate further evaluation recommended."
        elif malignant_prob >= 0.3:
            return "FOLLOW-UP: Moderate suspicion. Additional imaging or biopsy may be warranted."
        elif malignant_prob >= 0.1:
            return "MONITOR: Low suspicion but warrants continued surveillance."
        else:
            return "ROUTINE: Appears benign. Continue routine screening as appropriate."
    
    def _save_enhanced_model(self):
        """Save the enhanced model with additional metadata"""
        model_path = os.path.join(self.model_dir, "enhanced_mammography_classifier.pth")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': 'enhanced_mammography',
            'architecture': 'EfficientNet/ResNet50',
            'classes': self.classes,
            'is_trained': self.is_trained,
            'training_parameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            },
            'saved_at': datetime.now().isoformat()
        }, model_path)
        
        logger.info(f"Enhanced model saved to {model_path}")