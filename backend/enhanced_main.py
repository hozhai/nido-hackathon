"""
Enhanced Main API server with improved mammography model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging
import shutil
from datetime import datetime
import uuid
import torch

# Import both models for fallback
try:
    from improved_mammography_model import ImprovedMammographyClassifier
    ENHANCED_MODEL_AVAILABLE = True
    logger.info("Enhanced model available")
except ImportError as e:
    logger.warning(f"Enhanced model not available: {e}")
    ENHANCED_MODEL_AVAILABLE = False

from breast_cancer_model import BreastCancerClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Breast Cancer Mammography Detection API",
    description="AI-powered mammography analysis with improved accuracy for malignant detection",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Enhanced Mammography Classifier
logger.info("Initializing Enhanced Breast Cancer Mammography Detection System...")

# Set up paths relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "models")
data_dir = os.path.join(script_dir, "data")

# Try to load enhanced model first, fallback to original
if ENHANCED_MODEL_AVAILABLE:
    try:
        enhanced_model_path = os.path.join(model_dir, "enhanced_mammography_classifier.pth")
        if os.path.exists(enhanced_model_path):
            breast_cancer_classifier = ImprovedMammographyClassifier(model_dir=model_dir, data_dir=data_dir)
            breast_cancer_classifier.is_trained = True  # Mark as trained
            breast_cancer_classifier.model.load_state_dict(torch.load(enhanced_model_path)['model_state_dict'])
            logger.info("✅ Enhanced mammography model loaded successfully!")
            MODEL_TYPE = "enhanced"
        else:
            raise FileNotFoundError("Enhanced model not found")
    except Exception as e:
        logger.warning(f"Enhanced model loading failed: {e}. Using standard model.")
        breast_cancer_classifier = BreastCancerClassifier(model_dir=model_dir, data_dir=data_dir)
        MODEL_TYPE = "standard"
else:
    breast_cancer_classifier = BreastCancerClassifier(model_dir=model_dir, data_dir=data_dir)
    MODEL_TYPE = "standard"

# Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed extension"""
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.dcm', '.pgm'}
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)

@app.get("/")
async def root():
    return {
        "message": "Enhanced Breast Cancer Mammography Detection API",
        "version": "2.0.0",
        "model_type": MODEL_TYPE,
        "features": [
            "EfficientNet-B3 Architecture",
            "Focal Loss for Imbalanced Data",
            "Advanced Medical Augmentation",
            "Test-Time Augmentation",
            "Enhanced Risk Assessment"
        ] if MODEL_TYPE == "enhanced" else ["ResNet18 Standard Model"]
    }

@app.get("/model/status")
async def get_model_status():
    """Get the current status of the mammography model"""
    try:
        status_info = {
            "status": "success",
            "model_trained": breast_cancer_classifier.is_trained,
            "analysis_type": "mammography",
            "model_type": "Enhanced EfficientNet-B3" if MODEL_TYPE == "enhanced" else "ResNet18 (Breast Cancer Specialized)",
            "classes": breast_cancer_classifier.classes,
            "device": str(breast_cancer_classifier.device),
            "enhanced_features": MODEL_TYPE == "enhanced"
        }
        
        if hasattr(breast_cancer_classifier, 'training_accuracy') and breast_cancer_classifier.training_accuracy:
            status_info["training_accuracy"] = breast_cancer_classifier.training_accuracy
        if hasattr(breast_cancer_classifier, 'validation_accuracy') and breast_cancer_classifier.validation_accuracy:
            status_info["validation_accuracy"] = breast_cancer_classifier.validation_accuracy
            
        return status_info
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@app.post("/predict")
async def predict_mammography(image: UploadFile = File(...)):
    """
    Analyze mammography image for breast cancer detection
    Enhanced with improved accuracy for malignant detection
    """
    try:
        # Validate file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not allowed_file(image.filename):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload PNG, JPG, JPEG, TIFF, or DICOM images."
            )
        
        # Save uploaded file
        file_extension = os.path.splitext(image.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        logger.info(f"Mammography image uploaded: {file_path}")
        
        # Make prediction using enhanced model if available
        if MODEL_TYPE == "enhanced" and hasattr(breast_cancer_classifier, 'predict_with_confidence'):
            # Use enhanced prediction method
            result = breast_cancer_classifier.predict_with_confidence(file_path)
            
            # Format enhanced response
            response = {
                "status": "success",
                "filename": image.filename,
                "prediction": {
                    "class": result["predicted_class"],
                    "confidence": round(result["confidence"] * 100, 2),
                    "probabilities": {
                        "benign": round(result["probabilities"]["benign"] * 100, 2),
                        "malignant": round(result["probabilities"]["malignant"] * 100, 2)
                    }
                },
                "risk_assessment": {
                    "level": result["risk_level"],
                    "recommendation": result["recommendation"]
                },
                "model_info": {
                    "type": "Enhanced EfficientNet-B3",
                    "version": "2.0.0",
                    "features": result["technical_details"]
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Use standard prediction method
            prediction = breast_cancer_classifier.predict(file_path)
            
            response = {
                "status": "success",
                "filename": image.filename,
                "prediction": {
                    "class": prediction["predicted_class"],
                    "confidence": prediction["confidence"],
                    "probabilities": prediction.get("probabilities", {})
                },
                "model_info": {
                    "type": "ResNet18 Standard",
                    "version": "1.0.0"
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mammography analysis error: {str(e)}")
        # Clean up file on error
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": breast_cancer_classifier is not None,
        "model_trained": breast_cancer_classifier.is_trained if breast_cancer_classifier else False,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import torch  # Import here to avoid issues
    logger.info("🚀 Starting Enhanced Mammography Detection Server...")
    logger.info(f"📊 Using {MODEL_TYPE} model")
    logger.info("🔗 API available at: http://localhost:8000")
    logger.info("📖 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)