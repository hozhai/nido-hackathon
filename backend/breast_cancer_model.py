import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from typing import Dict
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreastCancerDataset(Dataset):
    """Custom dataset for breast cancer images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class BreastCancerClassifier:
    """
    Specialized breast cancer tissue classifier with automatic training
    """
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.classes = ["benign", "malignant"]  # Breast cancer classification
        self.is_trained = False
        self.confidence_threshold = 0.5
        self.training_accuracy = None
        self.validation_accuracy = None
        
        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "breast_cancer"), exist_ok=True)
        
        # Initialize transforms for image preprocessing
        self._setup_transforms()
        
        # Initialize the model architecture
        self._create_model_architecture()
        
        # Check if we have a trained model, if not download data and train
        model_path = os.path.join(self.model_dir, "breast_cancer_classifier.pth")
        if not os.path.exists(model_path):
            logger.info("No trained breast cancer model found. Starting automatic training...")
            self._download_and_prepare_breast_cancer_data()
            self._train_model()
        else:
            self._load_model(model_path)
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms for breast cancer histopathology"""
        # Training transforms with data augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation/inference transforms without augmentation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _create_model_architecture(self):
        """Create ResNet18 architecture specifically for breast cancer detection"""
        self.model = models.resnet18(pretrained=True)
        
        # Freeze early layers to retain general image features
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False
        
        # Modify the final layer for binary classification (benign/malignant)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # 2 classes: benign, malignant
        )
        
        self.model = self.model.to(self.device)
    
    def _download_and_prepare_breast_cancer_data(self):
        """Download and prepare breast cancer histopathology data"""
        breast_cancer_dir = os.path.join(self.data_dir, "breast_cancer")
        benign_dir = os.path.join(breast_cancer_dir, "benign")
        malignant_dir = os.path.join(breast_cancer_dir, "malignant")
        
        os.makedirs(benign_dir, exist_ok=True)
        os.makedirs(malignant_dir, exist_ok=True)
        
        # Create sample breast cancer data for training
        logger.info("Preparing breast cancer training dataset...")
        
        # Try to download real BreakHis dataset first
        if self._download_breakhis_dataset(breast_cancer_dir):
            logger.info("Real BreakHis dataset downloaded successfully")
            return
        
        # Fallback to synthetic data if real dataset not available
        logger.info("Real dataset unavailable, creating synthetic data for demonstration")
        self._create_synthetic_breast_cancer_data(benign_dir, malignant_dir)
        
        logger.info(f"Breast cancer dataset prepared in {breast_cancer_dir}")
    
    def _download_breakhis_dataset(self, output_dir: str) -> bool:
        """
        Download the BreakHis breast cancer histopathology dataset
        Returns True if successful, False if failed
        """
        try:
            import requests
            from zipfile import ZipFile
            
            # Note: This is a placeholder URL - BreakHis requires manual download
            # from https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
            
            logger.info("Checking for BreakHis dataset...")
            
            # Check if user has manually downloaded the dataset
            possible_paths = [
                os.path.join(output_dir, "BreaKHis_v1.tar.gz"),
                os.path.join(".", "BreaKHis_v1.tar.gz"),
                os.path.join("datasets", "BreaKHis_v1.tar.gz"),
            ]
            
            dataset_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_path = path
                    break
            
            if dataset_path:
                logger.info(f"Found BreakHis dataset at {dataset_path}")
                return self._extract_breakhis_dataset(dataset_path, output_dir)
            else:
                logger.warning("BreakHis dataset not found. Please download manually from:")
                logger.warning("https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/")
                logger.warning("Place BreaKHis_v1.tar.gz in the data directory")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process BreakHis dataset: {str(e)}")
            return False
    
    def _extract_breakhis_dataset(self, dataset_path: str, output_dir: str) -> bool:
        """Extract and organize the BreakHis dataset"""
        try:
            import tarfile
            import shutil
            
            logger.info("Extracting BreakHis dataset...")
            
            # Extract the tar.gz file
            with tarfile.open(dataset_path, 'r:gz') as tar:
                tar.extractall(output_dir)
            
            # Organize the data into benign/malignant folders
            breakhis_path = os.path.join(output_dir, "BreaKHis_v1", "histology_slides", "breast")
            benign_dir = os.path.join(output_dir, "benign")
            malignant_dir = os.path.join(output_dir, "malignant")
            
            os.makedirs(benign_dir, exist_ok=True)
            os.makedirs(malignant_dir, exist_ok=True)
            
            # Copy benign images
            benign_path = os.path.join(breakhis_path, "benign")
            if os.path.exists(benign_path):
                for root, dirs, files in os.walk(benign_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            src = os.path.join(root, file)
                            dst = os.path.join(benign_dir, file)
                            shutil.copy2(src, dst)
            
            # Copy malignant images
            malignant_path = os.path.join(breakhis_path, "malignant")
            if os.path.exists(malignant_path):
                for root, dirs, files in os.walk(malignant_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            src = os.path.join(root, file)
                            dst = os.path.join(malignant_dir, file)
                            shutil.copy2(src, dst)
            
            benign_count = len([f for f in os.listdir(benign_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            malignant_count = len([f for f in os.listdir(malignant_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            logger.info(f"Extracted {benign_count} benign and {malignant_count} malignant images")
            
            return benign_count > 0 and malignant_count > 0
            
        except Exception as e:
            logger.error(f"Failed to extract BreakHis dataset: {str(e)}")
            return False
    
    def _create_synthetic_breast_cancer_data(self, benign_dir: str, malignant_dir: str):
        """Create synthetic breast cancer images for training demonstration"""
        logger.info("Creating synthetic breast cancer training data...")
        
        # Create synthetic images that simulate breast cancer histopathology characteristics
        def create_synthetic_image(image_type: str, index: int, save_dir: str):
            """Create a synthetic histopathology-like image"""
            # Create base tissue-like pattern
            img = np.random.randint(180, 255, (224, 224, 3), dtype=np.uint8)
            
            if image_type == "malignant":
                # Add darker, more irregular patterns for malignant
                for _ in range(20):
                    x, y = np.random.randint(0, 200, 2)
                    size = np.random.randint(10, 30)
                    img[y:y+size, x:x+size] = np.random.randint(80, 150, (min(size, 224-y), min(size, 224-x), 3))
            else:
                # Add more regular, lighter patterns for benign
                for _ in range(15):
                    x, y = np.random.randint(0, 200, 2)
                    size = np.random.randint(5, 15)
                    img[y:y+size, x:x+size] = np.random.randint(200, 255, (min(size, 224-y), min(size, 224-x), 3))
            
            # Save image
            img_pil = Image.fromarray(img)
            img_path = os.path.join(save_dir, f"{image_type}_sample_{index:04d}.png")
            img_pil.save(img_path)
            return img_path
        
        # Create synthetic training data
        benign_images = []
        malignant_images = []
        
        # Generate benign samples
        for i in range(100):
            img_path = create_synthetic_image("benign", i, benign_dir)
            benign_images.append(img_path)
        
        # Generate malignant samples
        for i in range(100):
            img_path = create_synthetic_image("malignant", i, malignant_dir)
            malignant_images.append(img_path)
        
        logger.info(f"Created {len(benign_images)} benign and {len(malignant_images)} malignant synthetic samples")
        
        return benign_images, malignant_images
    
    def _load_breast_cancer_data(self):
        """Load breast cancer data from the prepared directory"""
        breast_cancer_dir = os.path.join(self.data_dir, "breast_cancer")
        benign_dir = os.path.join(breast_cancer_dir, "benign")
        malignant_dir = os.path.join(breast_cancer_dir, "malignant")
        
        image_paths = []
        labels = []
        
        # Load benign images
        if os.path.exists(benign_dir):
            for img_file in os.listdir(benign_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(benign_dir, img_file))
                    labels.append(0)  # 0 for benign
        
        # Load malignant images
        if os.path.exists(malignant_dir):
            for img_file in os.listdir(malignant_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(malignant_dir, img_file))
                    labels.append(1)  # 1 for malignant
        
        logger.info(f"Loaded {len(image_paths)} breast cancer images ({labels.count(0)} benign, {labels.count(1)} malignant)")
        
        return image_paths, labels
    
    def _train_model(self):
        """Train the breast cancer classification model"""
        logger.info("Starting breast cancer model training...")
        
        # Load training data
        image_paths, labels = self._load_breast_cancer_data()
        
        if len(image_paths) < 10:
            logger.warning("Insufficient training data. Creating additional synthetic data...")
            self._create_synthetic_breast_cancer_data(
                os.path.join(self.data_dir, "breast_cancer", "benign"),
                os.path.join(self.data_dir, "breast_cancer", "malignant")
            )
            image_paths, labels = self._load_breast_cancer_data()
        
        # Split data into training and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets and data loaders
        train_dataset = BreastCancerDataset(train_paths, train_labels, self.train_transform)
        val_dataset = BreastCancerDataset(val_paths, val_labels, self.transform)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        best_val_acc = 0.0
        epochs = 15
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels_batch in train_loader:
                images, labels_batch = images.to(self.device), labels_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images, labels_batch = images.to(self.device), labels_batch.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            logger.info(f'Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model()
            
            scheduler.step()
        
        self.training_accuracy = train_acc
        self.validation_accuracy = best_val_acc
        self.is_trained = True
        
        # Save final trained model
        self._save_model()
        
        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    def predict(self, image_path: str) -> Dict:
        """
        Predict breast cancer type (benign/malignant) with confidence score
        """
        logger.info(f"Prediction request - is_trained: {self.is_trained}, model: {self.model is not None}")
        
        if not self.is_trained or self.model is None:
            error_msg = f"Breast cancer model is not trained. is_trained: {self.is_trained}, model exists: {self.model is not None}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence_scores = probabilities.cpu().numpy()
                
                predicted_class_idx = np.argmax(confidence_scores)
                predicted_class = self.classes[predicted_class_idx]
                confidence = float(confidence_scores[predicted_class_idx] * 100)
                
                # Create probability dictionary
                prob_dict = {
                    class_name: float(conf * 100) 
                    for class_name, conf in zip(self.classes, confidence_scores)
                }
                
                # Risk assessment specific to breast cancer
                risk_level = self._assess_breast_cancer_risk(predicted_class, confidence)
                
                return {
                    "prediction": predicted_class,
                    "confidence": confidence,
                    "probabilities": prob_dict,
                    "risk_level": risk_level,
                    "interpretation": self._interpret_breast_cancer_result(predicted_class, confidence),
                    "cancer_type": "breast_cancer"
                }
                
        except Exception as e:
            logger.error(f"Breast cancer prediction failed: {str(e)}")
            raise e
    
    def _assess_breast_cancer_risk(self, prediction: str, confidence: float) -> str:
        """Assess breast cancer risk level based on prediction and confidence"""
        if prediction == "malignant":
            if confidence > 90:
                return "VERY_HIGH"
            elif confidence > 80:
                return "HIGH"
            elif confidence > 70:
                return "MODERATE_TO_HIGH"
            else:
                return "UNCERTAIN_MALIGNANT"
        else:  # benign
            if confidence > 90:
                return "VERY_LOW"
            elif confidence > 80:
                return "LOW"
            elif confidence > 70:
                return "LOW_TO_MODERATE"
            else:
                return "UNCERTAIN_BENIGN"
    
    def _interpret_breast_cancer_result(self, prediction: str, confidence: float) -> str:
        """Provide breast cancer-specific interpretation of the result"""
        if prediction == "malignant":
            if confidence > 90:
                return "High confidence detection of malignant breast tissue. Immediate oncological consultation strongly recommended."
            elif confidence > 80:
                return "Strong indication of malignant breast tissue detected. Urgent medical evaluation and biopsy confirmation advised."
            elif confidence > 70:
                return "Possible malignant breast tissue identified. Medical professional review and additional testing recommended."
            else:
                return "Uncertain classification with malignant tendency. Further imaging and expert pathology review needed."
        else:  # benign
            if confidence > 90:
                return "High confidence that breast tissue appears benign. Continue routine mammographic screening as advised by physician."
            elif confidence > 80:
                return "Breast tissue likely appears benign. Maintain regular screening schedule and follow physician guidance."
            elif confidence > 70:
                return "Breast tissue appears benign with moderate confidence. Consider follow-up imaging if clinically indicated."
            else:
                return "Uncertain classification with benign tendency. Additional imaging or tissue sampling may be warranted for definitive diagnosis."
    
    def _save_model(self):
        """Save the trained breast cancer model"""
        try:
            model_path = os.path.join(self.model_dir, "breast_cancer_classifier.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'classes': self.classes,
                'is_trained': self.is_trained,
                'training_accuracy': self.training_accuracy,
                'validation_accuracy': self.validation_accuracy,
                'cancer_type': 'breast_cancer',
                'saved_at': datetime.now().isoformat()
            }, model_path)
            logger.info(f"Breast cancer model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save breast cancer model: {str(e)}")
    
    def _load_model(self, model_path: str):
        """Load a saved breast cancer model"""
        try:
            logger.info(f"Loading breast cancer model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            self.is_trained = checkpoint.get('is_trained', True)
            self.training_accuracy = checkpoint.get('training_accuracy')
            self.validation_accuracy = checkpoint.get('validation_accuracy')
            
            logger.info(f"Breast cancer model loaded successfully - is_trained: {self.is_trained}")
            logger.info(f"Training accuracy: {self.training_accuracy}, Validation accuracy: {self.validation_accuracy}")
            
        except Exception as e:
            logger.error(f"Failed to load breast cancer model: {str(e)}")
            # If loading fails, retrain the model
            logger.info("Retraining model due to loading failure...")
            self.is_trained = False
            self._download_and_prepare_breast_cancer_data()
            self._train_model()
    
    def get_status(self) -> Dict:
        """Get current breast cancer model status"""
        return {
            "is_trained": self.is_trained,
            "model_type": "ResNet18 (Breast Cancer Specialized)",
            "cancer_type": "breast_cancer",
            "classes": self.classes,
            "device": str(self.device),
            "training_accuracy": self.training_accuracy,
            "validation_accuracy": self.validation_accuracy,
            "dataset_type": "Synthetic Breast Cancer Histopathology"
        }

# Utility functions for breast cancer dataset information
def get_breast_cancer_info():
    """Provide information about breast cancer detection"""
    info = """
    Breast Cancer Detection System Information:
    
    🎯 Specialization: Breast Cancer Histopathology
    📊 Classification: Benign vs Malignant breast tissue
    🧠 Model: ResNet18 with specialized architecture
    📈 Training: Automatic on-startup training with synthetic data
    
    Key Features:
    ✓ Breast cancer-specific risk assessment
    ✓ Medical interpretation tailored for breast pathology
    ✓ Confidence scoring optimized for histopathological analysis
    ✓ Automatic model training on initialization
    
    Real Dataset Integration:
    - BreakHis Dataset (for production use)
    - BACH Challenge Dataset
    - TCGA Breast Cancer Collections
    
    Note: Currently uses synthetic training data for demonstration.
    For production use, integrate with real breast cancer datasets.
    """
    return info