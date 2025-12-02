#!/bin/bash
# Optional script to install pyserini (for custom retrieval generation)
# This may fail due to spacy build issues - that's OK if you're using pre-computed results

set -e

echo "Attempting to install pyserini..."

# Activate venv if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "Error: Virtual environment not found. Run ./setup_venv.sh first."
        exit 1
    fi
fi

# Try to install spacy first (often helps with build issues)
echo "Installing spacy first..."
pip install spacy || echo "Warning: spacy installation failed, continuing anyway..."

# Install pyserini
echo "Installing pyserini..."
pip install pyserini || {
    echo ""
    echo "⚠ pyserini installation failed. This is OK if you're using pre-computed top-1000 results."
    echo "The passage reranking experiment will work fine without pyserini."
    echo ""
    echo "If you need pyserini, you can try:"
    echo "  1. Install build tools: xcode-select --install (macOS)"
    echo "  2. Install spacy separately: pip install spacy"
    echo "  3. Then retry: pip install pyserini"
    exit 0
}

echo "✓ pyserini installed successfully!"

