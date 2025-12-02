#!/bin/bash
# Setup script for Python virtual environment for passage reranking experiments

set -e

echo "Setting up Python virtual environment..."

# Check if python3.10 is available
if ! command -v python3.10 &> /dev/null; then
    echo "Error: python3.10 not found. Please install Python 3.10."
    echo "On macOS, you can install it with: brew install python@3.10"
    exit 1
fi

# Create virtual environment with Python 3.10
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."

# Install spacy first (pyserini dependency) - try multiple methods
echo "Installing spacy (required for pyserini)..."
if ! pip install "spacy>=3.4.0,<3.8.0" 2>/dev/null; then
    echo "Trying spacy installation with --no-build-isolation..."
    pip install spacy --no-build-isolation || {
        echo "Warning: spacy installation failed. Trying with pre-built wheel..."
        pip install --only-binary :all: spacy || {
            echo "Error: Could not install spacy. This is required for pyserini."
            echo "Try: pip install spacy --no-cache-dir"
            exit 1
        }
    }
fi

echo "Installing remaining requirements..."
pip install -r requirements.txt

# Install permsc package in development mode
echo "Installing permsc package in development mode..."
pip install -e .

# Download NLTK data (required for some datasets)
python -c "import nltk; nltk.download('punkt', quiet=True)"

echo ""
echo "Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"

