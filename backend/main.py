from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from typing import List
import logging

# Import custom modules
from ml_model import ImageClassifier
from utils import save_upload_file, allowed_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Image Classification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Nuxt dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML model
classifier = ImageClassifier()

# Configuration
UPLOAD_DIR = "uploads"
MODEL_DIR = "models"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Image Classification API is running"}

@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    """
    Predict the class of an uploaded image
    """
    try:
        # Validate file
        if not allowed_file(image.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Save uploaded file
        file_path = await save_upload_file(image, UPLOAD_DIR)
        logger.info(f"Image uploaded: {file_path}")
        
        # Make prediction
        prediction_result = classifier.predict(file_path)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return JSONResponse(content={
            "status": "success",
            "prediction": prediction_result["class"],
            "confidence": prediction_result["confidence"],
            "probabilities": prediction_result.get("probabilities", {})
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/train")
async def train_model(
    images: List[UploadFile] = File(...),
    labels: List[str] = None
):
    """
    Train the model with uploaded images and labels
    """
    try:
        if len(images) < 2:
            raise HTTPException(status_code=400, detail="At least 2 images required for training")
        
        # Save uploaded files
        image_paths = []
        for image in images:
            if not allowed_file(image.filename):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {image.filename}")
            
            file_path = await save_upload_file(image, UPLOAD_DIR)
            image_paths.append(file_path)
        
        logger.info(f"Starting training with {len(image_paths)} images")
        
        # Train the model
        training_result = classifier.train(image_paths, labels)
        
        # Clean up uploaded files
        for file_path in image_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Model trained successfully",
            "accuracy": training_result.get("accuracy"),
            "classes": training_result.get("classes", [])
        })
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/model/status")
async def get_model_status():
    """
    Get the current model status and information
    """
    try:
        status = classifier.get_status()
        return JSONResponse(content={
            "status": "success",
            "model_trained": status["is_trained"],
            "classes": status.get("classes", []),
            "accuracy": status.get("accuracy"),
            "last_trained": status.get("last_trained")
        })
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
