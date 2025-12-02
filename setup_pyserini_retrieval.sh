#!/bin/bash
# Script to set up Pyserini retrieval for MS MARCO passage reranking

set -e

echo "Setting up Pyserini retrieval for MS MARCO..."

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment not activated. Run 'source venv/bin/activate' first."
    exit 1
fi

# Check if pyserini is installed
if ! python -c "import pyserini" 2>/dev/null; then
    echo "Error: pyserini not installed. Run 'pip install pyserini' first."
    exit 1
fi

# Generate retrieval results for DL19
echo ""
echo "Generating BM25 retrieval results for TREC DL19..."
python generate_retrieval_results.py \
    --collection data/msmarco/collection.tsv \
    --queries data/trec-dl19/msmarco-test2019-queries.tsv \
    --index indexes/msmarco-passage \
    --output data/trec-dl19/bm25-top100.tsv \
    --k 100

# Generate retrieval results for DL20
echo ""
echo "Generating BM25 retrieval results for TREC DL20..."
python generate_retrieval_results.py \
    --collection data/msmarco/collection.tsv \
    --queries data/trec-dl20/msmarco-test2020-queries.tsv \
    --index indexes/msmarco-passage \
    --output data/trec-dl20/bm25-top100.tsv \
    --k 100 \
    --skip-index  # Reuse index from DL19

echo ""
echo "✓ Retrieval results generated!"
echo ""
echo "Files created:"
echo "  - data/trec-dl19/bm25-top100.tsv"
echo "  - data/trec-dl20/bm25-top100.tsv"
echo ""
echo "You can now use these files in the notebook by setting:"
echo "  top1000_path='data/trec-dl19/bm25-top100.tsv'  # or dl20"

