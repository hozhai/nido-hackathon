from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from typing import List
import logging

# Import custom modules
from breast_cancer_model import BreastCancerClassifier, get_breast_cancer_info
from utils import save_upload_file, allowed_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Breast Cancer Mammography Detection API", version="1.0.0", description="AI-powered breast cancer mammography analysis")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Nuxt dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Breast Cancer Mammography Classifier (will auto-train on first startup)
logger.info("Initializing Breast Cancer Mammography Detection System...")

# Set up paths relative to this script's location
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "models")
data_dir = os.path.join(script_dir, "data")

breast_cancer_classifier = BreastCancerClassifier(model_dir=model_dir, data_dir=data_dir)

# Configuration
UPLOAD_DIR = "uploads"
MODEL_DIR = "models"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Breast Cancer Mammography Detection API is running", "status": "healthy", "version": "1.0.0"}

@app.post("/predict")
async def predict_breast_cancer(image: UploadFile = File(...)):
    """
    Predict breast cancer from mammography images
    Analyzes mammographic findings for suspicious patterns
    """
    try:
        # Validate file
        if image.filename and not allowed_file(image.filename):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload PNG, JPG, JPEG, TIFF, or DICOM images.")
        
        # Save uploaded file
        file_path = await save_upload_file(image, UPLOAD_DIR)
        logger.info(f"Mammography image uploaded: {file_path}")
        
        # Make prediction using the breast cancer mammography classifier
        prediction_result = breast_cancer_classifier.predict(file_path)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return JSONResponse(content={
            "status": "success",
            "analysis_type": "mammography",
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "probabilities": prediction_result["probabilities"],
            "risk_level": prediction_result["risk_level"],
            "interpretation": prediction_result["interpretation"],
            "model_info": "Specialized Breast Cancer Mammography Detection Model"
        })
        
    except Exception as e:
        logger.error(f"Mammography analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Mammography analysis failed: {str(e)}")

@app.get("/model/info")
async def get_breast_cancer_info_endpoint():
    """
    Get information about the breast cancer mammography detection system
    """
    try:
        return JSONResponse(content={
            "status": "success",
            "system_info": get_breast_cancer_info(),
            "model_status": breast_cancer_classifier.get_status()
        })
    except Exception as e:
        logger.error(f"Info retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")

@app.get("/model/status")
async def get_model_status():
    """
    Get the current breast cancer mammography model status and performance metrics
    """
    try:
        status = breast_cancer_classifier.get_status()
        return JSONResponse(content={
            "status": "success",
            "model_trained": status["is_trained"],
            "analysis_type": "mammography",
            "model_type": status["model_type"],
            "classes": status.get("classes", []),
            "training_accuracy": status.get("training_accuracy"),
            "validation_accuracy": status.get("validation_accuracy"),
            "device": status.get("device"),
            "dataset_type": status.get("dataset_type")
        })
    except Exception as e:
        logger.error(f"Status retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
