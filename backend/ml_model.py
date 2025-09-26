import os
import joblib
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Image processing imports
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageClassifier:
    """
    Image classifier using scikit-learn with feature extraction
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.classes = []
        self.accuracy = None
        self.last_trained = None
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from an image using various techniques
        """
        try:
            # Load image using PIL
            image = Image.open(image_path).convert('RGB')
            
            # Resize to standard size
            image = image.resize((64, 64))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Extract basic color features
            features = []
            
            # Color histogram features
            for channel in range(3):  # RGB channels
                hist = cv2.calcHist([image_array], [channel], None, [8], [0, 256])
                features.extend(hist.flatten())
            
            # Basic statistical features
            features.extend([
                np.mean(image_array),
                np.std(image_array),
                np.min(image_array),
                np.max(image_array)
            ])
            
            # Texture features (using simple edge detection)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            features.extend([
                np.mean(edges),
                np.sum(edges > 0) / edges.size  # Edge density
            ])
            
            # Shape features
            features.extend([
                image_array.shape[0] * image_array.shape[1],  # Area
                image_array.shape[0] / image_array.shape[1]   # Aspect ratio
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {str(e)}")
            # Return zero features if extraction fails
            return np.zeros(28, dtype=np.float32)  # Adjust size based on feature count
    
    def train(self, image_paths: List[str], labels: Optional[List[str]] = None) -> Dict:
        """
        Train the classifier with the provided images and labels
        """
        try:
            logger.info(f"Starting training with {len(image_paths)} images")
            
            # Extract features from all images
            features = []
            actual_labels = []
            
            for i, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                feature_vector = self.extract_features(image_path)
                features.append(feature_vector)
                
                # Use provided labels or generate based on filename/directory
                if labels and i < len(labels):
                    actual_labels.append(labels[i])
                else:
                    # Extract label from filename or use generic labels
                    label = self._extract_label_from_filename(image_path)
                    actual_labels.append(label)
            
            if len(features) < 2:
                raise ValueError("Need at least 2 samples for training")
            
            X = np.array(features)
            y = np.array(actual_labels)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            self.classes = self.label_encoder.classes_.tolist()
            
            # Split data for validation
            if len(X) > 4:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
            else:
                X_train, X_test = X, X
                y_train, y_test = y_encoded, y_encoded
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=2
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            self.accuracy = accuracy_score(y_test, y_pred)
            
            # Update training status
            self.is_trained = True
            self.last_trained = datetime.now().isoformat()
            
            # Save model
            self._save_model()
            
            logger.info(f"Training completed. Accuracy: {self.accuracy:.2f}")
            
            return {
                "accuracy": self.accuracy,
                "classes": self.classes,
                "n_samples": len(X),
                "training_date": self.last_trained
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise e
    
    def predict(self, image_path: str) -> Dict:
        """
        Predict the class of a single image
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained. Please train the model first.")
        
        try:
            # Extract features
            features = self.extract_features(image_path)
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get class name
            class_name = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get confidence (max probability)
            confidence = float(np.max(probabilities) * 100)
            
            # Create probabilities dictionary
            prob_dict = {
                class_name: float(prob * 100) 
                for class_name, prob in zip(self.classes, probabilities)
            }
            
            return {
                "class": class_name,
                "confidence": confidence,
                "probabilities": prob_dict
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise e
    
    def get_status(self) -> Dict:
        """
        Get the current status of the model
        """
        return {
            "is_trained": self.is_trained,
            "classes": self.classes,
            "accuracy": self.accuracy,
            "last_trained": self.last_trained
        }
    
    def _extract_label_from_filename(self, image_path: str) -> str:
        """
        Extract label from image filename or directory structure
        """
        filename = os.path.basename(image_path)
        # Remove extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Try to extract meaningful label from filename
        # This is a simple heuristic - can be improved based on naming convention
        parts = name_without_ext.split('_')
        if len(parts) > 1:
            return parts[0].lower()
        
        # Check parent directory
        parent_dir = os.path.basename(os.path.dirname(image_path))
        if parent_dir and parent_dir != "uploads":
            return parent_dir.lower()
        
        # Default fallback
        return "unknown"
    
    def _save_model(self) -> None:
        """
        Save the trained model and associated components
        """
        try:
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "classes": self.classes,
                "accuracy": self.accuracy,
                "last_trained": self.last_trained,
                "is_trained": self.is_trained
            }
            
            model_path = os.path.join(self.model_dir, "classifier_model.joblib")
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
    
    def _load_model(self) -> None:
        """
        Load a previously trained model
        """
        try:
            model_path = os.path.join(self.model_dir, "classifier_model.joblib")
            
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                
                self.model = model_data.get("model")
                self.scaler = model_data.get("scaler", StandardScaler())
                self.label_encoder = model_data.get("label_encoder", LabelEncoder())
                self.classes = model_data.get("classes", [])
                self.accuracy = model_data.get("accuracy")
                self.last_trained = model_data.get("last_trained")
                self.is_trained = model_data.get("is_trained", False)
                
                if self.is_trained:
                    logger.info(f"Loaded existing model with {len(self.classes)} classes")
                
        except Exception as e:
            logger.warning(f"Could not load existing model: {str(e)}")
            # Initialize with default values if loading fails
            self.model = None
            self.is_trained = False
