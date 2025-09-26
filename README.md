# Image Classifier - Full Stack Application

A full-stack image classification application with a Nuxt.js frontend and Python FastAPI backend using scikit-learn for machine learning.

## Project Structure

```
nido-hackathon/
├── frontend/                 # Nuxt.js application
│   ├── components/          # Vue components
│   ├── pages/              # Application pages
│   ├── assets/             # Static assets
│   └── package.json        # Frontend dependencies
├── backend/                 # Python FastAPI application
│   ├── main.py             # FastAPI application entry point
│   ├── ml_model.py         # Machine learning model implementation
│   ├── utils.py            # Utility functions
│   ├── requirements.txt    # Python dependencies
│   ├── models/             # Trained model storage
│   └── uploads/            # Temporary file storage
├── shared/                  # Shared utilities and types
└── docker-compose.yml      # Docker orchestration
```

## Features

### Frontend (Nuxt.js)

- Image upload interface with drag & drop support
- Real-time prediction results display
- Model training interface
- Responsive design with Tailwind CSS
- File validation and preview

### Backend (Python FastAPI)

- RESTful API endpoints for image processing
- Machine learning model training with scikit-learn
- Image feature extraction using OpenCV and PIL
- File upload handling with validation
- Model persistence and loading
- CORS configuration for frontend integration

### Machine Learning

- Random Forest classifier for image classification
- Feature extraction from images (color histograms, texture, shape)
- Model training with user-uploaded datasets
- Prediction confidence scoring
- Model status and information endpoints

## Getting Started

### Prerequisites

- Node.js 18+ (for frontend)
- Python 3.11+ (for backend)
- Docker and Docker Compose (for containerized deployment)

### Development Setup

#### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Configure environment variables
uvicorn main:app --reload
```

#### Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env  # Configure environment variables
npm run dev
```

### Docker Deployment

```bash
# Build and run both services
docker-compose up --build

# Run in detached mode
docker-compose up -d
```

## API Endpoints

### Backend API (http://localhost:8000)

- `GET /` - Health check
- `POST /predict` - Predict image class
- `POST /train` - Train model with uploaded images
- `GET /model/status` - Get model status and information

### Frontend Application (http://localhost:3000)

- `/` - Main application interface with upload and prediction

## Usage

1. **Training the Model**:

   - Upload multiple images using the training section
   - Provide labels for your images (or use automatic labeling)
   - Click "Train Model" to start the training process

2. **Making Predictions**:
   - Upload an image using the prediction section
   - Click "Predict" to classify the image
   - View results with confidence scores

## Development Notes

This is a scaffolded project with stub implementations. The following areas need full implementation:

- **Frontend**: Complete API integration, error handling, and styling
- **Backend**: Enhanced error handling, authentication, and logging
- **ML Model**: Advanced feature extraction, model optimization, and evaluation metrics
- **Infrastructure**: Production deployment configuration, monitoring, and scaling

## Environment Variables

### Backend (.env)

```
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
FRONTEND_URL=http://localhost:3000
MAX_FILE_SIZE=10485760
```

### Frontend (.env)

```
API_BASE_URL=http://localhost:8000
APP_NAME="Image Classifier"
MAX_FILE_SIZE=10485760
```

## Technologies Used

### Frontend

- Nuxt.js 3
- Vue.js 3
- Tailwind CSS
- @nuxt/ui

### Backend

- FastAPI
- scikit-learn
- OpenCV
- Pillow (PIL)
- NumPy
- Uvicorn

### DevOps

- Docker & Docker Compose
- Git

## License

This project is licensed under the ISC License.
