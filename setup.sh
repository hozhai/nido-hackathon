#!/bin/bash

# Development setup script for the Image Classifier project

echo "🚀 Setting up Image Classifier Development Environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+ and try again."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ and try again."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Setup backend
echo "🐍 Setting up Python backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Created Python virtual environment"
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
echo "✅ Installed Python dependencies"

# Copy environment file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env file for backend"
fi

cd ..

# Setup frontend
echo "📦 Setting up Node.js frontend..."
cd frontend

# Install npm dependencies
npm install
echo "✅ Installed Node.js dependencies"

# Copy environment file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env file for frontend"
fi

cd ..

echo ""
echo "🎉 Setup complete! You can now start the development servers:"
echo ""
echo "Backend (from backend/ directory):"
echo "  source venv/bin/activate"
echo "  uvicorn main:app --reload"
echo ""
echo "Frontend (from frontend/ directory):"
echo "  npm run dev"
echo ""
echo "Or use Docker Compose:"
echo "  docker-compose up --build"
echo ""
echo "The application will be available at:"
echo "  Frontend: http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
