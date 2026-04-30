#!/bin/bash

echo "============================================================"
echo "Smart File AI Chatbot - Linux/Mac Quick Start"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Found Python $PYTHON_VERSION"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "WARNING: Ollama is not running"
    echo "Please start Ollama:"
    echo "1. Install from https://ollama.ai"
    echo "2. Run: ollama pull llama2"
    echo "3. Run: ollama serve (in background terminal)"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Navigate to backend directory
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Start the server
echo ""
echo "============================================================"
echo "Starting FastAPI Server..."
echo "============================================================"
echo ""
echo "Backend will be available at: http://localhost:8000"
echo "API Docs at: http://localhost:8000/docs"
echo ""
echo "In another terminal, run:"
echo "cd frontend"
echo "python3 -m http.server 8080"
echo ""
echo "Then open: http://localhost:8080"
echo ""

python3 main.py
