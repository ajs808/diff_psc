# Passage Reranking Experiment Setup

This document provides setup instructions for running the passage reranking experiments.

## Quick Start

### 1. Set Up Virtual Environment

```bash
# Run the setup script
./setup_venv.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Dataset Loading

Test that the dataset loads correctly:

```python
from permsc import MSMarcoDataset

ds = MSMarcoDataset(
    collection_path='data/msmarco/collection.tsv',
    queries_path='data/trec-dl19/msmarco-test2019-queries.tsv',
    qrels_path='data/trec-dl19/2019qrels-pass.txt',
    top1000_path='data/trec-dl19/msmarco-passagetest2019-top1000.tsv',
    max_passages=100
)

print(f"Loaded {len(ds)} queries")
example = ds[0]
print(f"Query: {example.query.content}")
print(f"Number of passages: {len(example.hits)}")
```

### 3. Run the Notebook

1. Activate the virtual environment: `source venv/bin/activate`
2. Start Jupyter: `jupyter notebook`
3. Open `notebooks/passage-reranking.ipynb`
4. Set your OpenAI API key in the first code cell
5. Run all cells

## Dataset Notes

### Using Pre-computed Top-1000 Results

The notebook uses the pre-computed top-1000 retrieval results from the TREC datasets:
- `data/trec-dl19/msmarco-passagetest2019-top1000.tsv`
- `data/trec-dl20/msmarco-passagetest2020-top1000.tsv`

These files contain the top-1000 passages per query from BM25 retrieval. The dataset loader automatically limits to top-100 (configurable via `max_passages` parameter).

### Using Pyserini for Custom Retrieval

If you want to generate your own retrieval results using Pyserini:

1. **Index the MS MARCO collection:**
   ```bash
   python -m pyserini.index \
       -collection JsonCollection \
       -generator DefaultLuceneDocumentGenerator \
       -threads 9 \
       -input data/msmarco/collection_json \
       -index indexes/msmarco-passage \
       -storePositions -storeDocvectors -storeRaw
   ```

2. **Retrieve top-100 for queries:**
   ```bash
   python -m pyserini.search \
       --index indexes/msmarco-passage \
       --topics data/trec-dl19/msmarco-test2019-queries.tsv \
       --output data/trec-dl19/bm25-top100.tsv \
       --bm25 \
       --hits 100
   ```

3. **Modify the dataset loader** to use your custom retrieval results.

**Note:** The current implementation uses the pre-computed top-1000 files, so Pyserini is not strictly required unless you want to generate custom retrieval results.

## Configuration

### In the Notebook

- `api_key`: Your OpenAI API key
- `api_type`: 'openai' or 'azure'
- `num_aggregates`: Number of permutations (paper uses 20)
- `num_limit`: Number of queries to process (200 for full DL19/DL20)
- `track`: 'dl19' or 'dl20'

### Dataset Parameters

- `max_passages`: Maximum passages per query (default: 100)
- `use_passage_from_top1000`: Use passage text from top1000 file (faster) or look up from collection

## Evaluation Metrics

The experiment evaluates using:
- **nDCG@10**: Normalized Discounted Cumulative Gain at rank 10
- **MRR@10**: Mean Reciprocal Rank at rank 10

These metrics are computed against the TREC relevance judgments (qrels).

## Troubleshooting

### Import Errors
Make sure the virtual environment is activated and requirements are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Dataset Loading Errors
Verify all dataset files are in the correct locations (see `DATASET_DOWNLOAD_GUIDE.md`).

### API Errors
- Check your OpenAI API key is set correctly
- Verify you have sufficient API credits
- For Azure, check the `api_type` and endpoint configuration

## File Structure

```
diff_psc/
├── data/
│   ├── msmarco/
│   │   └── collection.tsv
│   ├── trec-dl19/
│   │   ├── msmarco-test2019-queries.tsv
│   │   ├── msmarco-passagetest2019-top1000.tsv
│   │   └── 2019qrels-pass.txt
│   └── trec-dl20/
│       ├── msmarco-test2020-queries.tsv
│       ├── msmarco-passagetest2020-top1000.tsv
│       └── 2020qrels-pass.txt
├── notebooks/
│   └── passage-reranking.ipynb
├── permsc/
│   ├── data.py (contains MSMarcoDataset)
│   └── evaluation.py (contains nDCG@10, MRR@10)
└── requirements.txt

```

