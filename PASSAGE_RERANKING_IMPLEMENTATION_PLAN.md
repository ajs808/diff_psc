# Passage Reranking Experiment Implementation Plan

## Overview
This document outlines the concrete steps needed to implement the passage reranking experiment from the paper "Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models" (arXiv:2310.07712).

## Current State Analysis

### What Already Exists
1. **Core Infrastructure:**
   - `RelevanceRankingPromptBuilder` class in `permsc/llm/prompt_builder.py` - already designed for passage reranking (uses RankGPT-style prompts)
   - `RankingExample` class with query support (`query` field)
   - Aggregation pipeline (`KemenyOptimalAggregator`, `BordaRankAggregator`, etc.)
   - LLM pipeline infrastructure (`OpenAIPromptPipeline`, `ChatCompletionPool`)
   - Permutation handling in `RankingExample.randomize_order()`

2. **Similar Implementation:**
   - Sorting tasks notebook (`notebooks/sorting-tasks.ipynb`) provides a template for the experiment structure

### What's Missing
1. MS MARCO dataset loader
2. TREC DL19/DL20 dataset integration
3. First-stage retrieval setup
4. Evaluation metrics (nDCG@10, MRR@10)
5. Passage reranking experiment notebook

---

## Implementation Steps

### Step 1: Dataset Acquisition and Understanding

#### 1.1 MS MARCO v1 Corpus
- **What:** 8.8 million passages
- **Where to get it:**
  - Official MS MARCO website: https://microsoft.github.io/msmarco/
  - Direct download: `collection.tar.gz` (contains `collection.tsv`)
  - Format: TSV with columns: `pid` (passage ID), `passage` (text content)

#### 1.2 TREC Deep Learning Tracks
- **What:** Queries and relevance judgments (qrels)
- **Where to get it:**
  - **TREC DL19 (2019):**
    - Queries: https://trec.nist.gov/data/deep2019q/ (queries.tsv)
    - Qrels: https://trec.nist.gov/data/deep2019q/qrels-pass.txt
  - **TREC DL20 (2020):**
    - Queries: https://trec.nist.gov/data/deep2020q/ (queries.tsv)
    - Qrels: https://trec.nist.gov/data/deep2020q/qrels-pass.txt
- **Format:**
  - Queries: TSV with `qid` and `query` columns
  - Qrels: TREC format: `qid Q0 pid relevance`

#### 1.3 First-Stage Retrieval Results
- **What:** Top-100 passages per query from first-stage retrieval
- **Options:**
  - **BM25:** Use Pyserini or Anserini to generate BM25 rankings
  - **SPLADE++ ED:** Use pre-computed rankings if available, or run SPLADE++ EnsembleDistill
- **Format needed:** For each query, a list of top-100 passage IDs with scores
- **Note:** The paper uses both BM25 and SPLADE++ ED as baselines

---

### Step 2: Create MS MARCO Dataset Loader

#### 2.1 Implement `MSMarcoDataset` class
- **Location:** `permsc/data.py`
- **Structure:** Extend `RankingDataset` base class
- **Requirements:**
  - Load MS MARCO collection (passage ID → passage text mapping)
  - Load TREC queries (query ID → query text)
  - Load TREC qrels (query ID → relevant passage IDs with relevance scores)
  - Load first-stage retrieval results (query ID → list of top-100 passage IDs)
  - Implement `load_example(idx)` that returns a `RankingExample`:
    - `query`: `Item` with query text
    - `hits`: List of `Item` objects, each containing:
      - `content`: passage text
      - `id`: passage ID
      - `score`: first-stage retrieval score (optional)
      - `metadata`: relevance label from qrels (if available)

#### 2.2 Data Preprocessing Considerations
- Handle missing passages (if passage ID not found in collection)
- Handle queries with fewer than 100 retrieved passages
- Store relevance labels in metadata for evaluation
- Support filtering to only queries that have relevance judgments

---

### Step 3: Set Up First-Stage Retrieval (Optional but Recommended)

#### 3.1 Option A: Use Pre-computed Rankings
- Download pre-computed BM25 or SPLADE++ rankings if available
- Format: JSON/TSV with query ID → list of (passage ID, score) tuples

#### 3.2 Option B: Generate Rankings with Pyserini
- Install Pyserini: `pip install pyserini`
- Index MS MARCO collection: `python -m pyserini.index -collection JsonCollection ...`
- Retrieve top-100 for each query: `python -m pyserini.search ...`
- Save results in format compatible with `MSMarcoDataset`

#### 3.3 Integration
- Modify `MSMarcoDataset` to accept path to retrieval results file
- Load retrieval results and map to queries

---

### Step 4: Implement Evaluation Metrics

#### 4.1 Create Evaluation Module
- **Location:** `permsc/evaluation.py` (new file)
- **Functions needed:**
  - `ndcg_at_k(ranked_passages, qrels, k=10)`: Compute nDCG@k
    - Input: ranked list of passage IDs, qrels dict (query_id → {passage_id: relevance})
    - Output: nDCG@k score
  - `mrr_at_k(ranked_passages, qrels, k=10)`: Compute MRR@k
    - Input: ranked list of passage IDs, qrels dict
    - Output: MRR@k score
  - `recall_at_k(ranked_passages, qrels, k=10)`: Compute Recall@k (optional)

#### 4.2 Implementation Details
- Use standard IR evaluation formulas
- Handle missing relevance judgments (treat as 0)
- Support multiple relevance levels (binary or graded)

---

### Step 5: Create Passage Reranking Experiment Notebook

#### 5.1 Notebook Structure (`notebooks/passage-reranking.ipynb`)
Follow the pattern from `sorting-tasks.ipynb`:

**Cell 1: Setup and Configuration**
```python
api_key = ''
api_type = 'openai'  # or 'azure'
num_aggregates = 20  # Paper uses 20 permutations
num_limit = 100  # Number of queries to process (for testing)
dataset_name = 'dl19'  # or 'dl20'
retrieval_method = 'bm25'  # or 'splade'
```

**Cell 2: Import and Initialize**
```python
from permsc import (
    RelevanceRankingPromptBuilder, 
    OpenAIPromptPipeline, 
    OpenAIConfig, 
    ChatCompletionPool,
    KemenyOptimalAggregator,
    MSMarcoDataset  # to be created
)
from permsc.evaluation import ndcg_at_k, mrr_at_k  # to be created

config = OpenAIConfig(model_name='gpt-3.5-turbo', api_key=api_key, api_type=api_type)
builder = RelevanceRankingPromptBuilder()
pool = ChatCompletionPool([config] * 5)  # 5 parallel instances
pipeline = OpenAIPromptPipeline(builder, pool)
```

**Cell 3: Load Dataset**
```python
# Paths to data files
collection_path = '../data/msmarco/collection.tsv'
queries_path = '../data/trec-dl19/queries.tsv'
qrels_path = '../data/trec-dl19/qrels-pass.txt'
retrieval_results_path = '../data/trec-dl19/bm25-top100.tsv'

ds = MSMarcoDataset(
    collection_path=collection_path,
    queries_path=queries_path,
    qrels_path=qrels_path,
    retrieval_results_path=retrieval_results_path
)
```

**Cell 4: Run Pipeline Function**
```python
def run_passage_reranking_pipeline(pipeline, dataset, num_aggregates, limit=100):
    """
    Similar to run_pipeline in sorting-tasks.ipynb, but adapted for passage reranking.
    Returns: prefs_list, perms_list, qrels_dict
    """
    prefs_list = []
    perms_list = []
    qrels_dict = {}  # Store qrels for evaluation
    
    for example in dataset[:limit]:
        example = deepcopy(example)
        query_id = example.metadata.get('query_id')
        qrels_dict[query_id] = example.metadata.get('qrels', {})
        
        prefs = []
        items = []
        perms = []
        
        # Generate num_aggregates permutations
        for _ in range(num_aggregates):
            ex_cpy = deepcopy(example)
            perm = ex_cpy.randomize_order()
            perms.append(perm)
            items.append(ex_cpy)
        
        # Run pipeline
        outputs = pipeline.run(items, temperature=0, request_timeout=30)
        
        # Restore preferences to original order
        for output, perm in zip(outputs, perms):
            restored_prefs = example.permuted_preferences_to_original_order(output)
            prefs.append(restored_prefs)
        
        prefs_list.append(np.array(prefs))
        perms_list.append(np.array(perms))
    
    return prefs_list, perms_list, qrels_dict
```

**Cell 5: Aggregate Rankings**
```python
def aggregate_rankings(prefs_list):
    aggregator = KemenyOptimalAggregator()
    results = []
    
    for prefs in prefs_list:
        aggregated = aggregator.aggregate(prefs)
        results.append(aggregated)
    
    return results
```

**Cell 6: Evaluate Results**
```python
def evaluate_reranking(results, dataset, qrels_dict):
    """
    Evaluate aggregated rankings using nDCG@10 and MRR@10
    """
    ndcg_scores = []
    mrr_scores = []
    
    for idx, (result, example) in enumerate(zip(results, dataset)):
        query_id = example.metadata.get('query_id')
        qrels = qrels_dict.get(query_id, {})
        
        # Convert preference array to ranked passage IDs
        ranked_passage_ids = [example.hits[i].id for i in result if i != -1]
        
        # Compute metrics
        ndcg = ndcg_at_k(ranked_passage_ids, qrels, k=10)
        mrr = mrr_at_k(ranked_passage_ids, qrels, k=10)
        
        ndcg_scores.append(ndcg)
        mrr_scores.append(mrr)
    
    return {
        'ndcg@10': np.mean(ndcg_scores),
        'mrr@10': np.mean(mrr_scores),
        'ndcg_scores': ndcg_scores,
        'mrr_scores': mrr_scores
    }
```

**Cell 7: Run Experiment**
```python
# Run pipeline
prefs_list, perms_list, qrels_dict = run_passage_reranking_pipeline(
    pipeline, ds, num_aggregates, limit=num_limit
)

# Aggregate
results = aggregate_rankings(prefs_list)

# Evaluate
metrics = evaluate_reranking(results, ds[:num_limit], qrels_dict)
print(f"nDCG@10: {metrics['ndcg@10']:.4f}")
print(f"MRR@10: {metrics['mrr@10']:.4f}")
```

**Cell 8: Compare with Baseline**
```python
# Evaluate first-stage retrieval baseline
baseline_results = []
for example in ds[:num_limit]:
    # Use original retrieval order
    ranked_ids = [hit.id for hit in example.hits]
    baseline_results.append(ranked_ids)

baseline_metrics = evaluate_reranking(baseline_results, ds[:num_limit], qrels_dict)
print(f"Baseline nDCG@10: {baseline_metrics['ndcg@10']:.4f}")
print(f"Baseline MRR@10: {baseline_metrics['mrr@10']:.4f}")
```

---

### Step 6: Code Implementation Details

#### 6.1 MSMarcoDataset Implementation
```python
class MSMarcoDataset(RankingDataset):
    def __init__(
        self, 
        collection_path: str,
        queries_path: str,
        qrels_path: str,
        retrieval_results_path: str,
        max_passages: int = 100
    ):
        # Load collection (pid -> passage text)
        self.collection = self._load_collection(collection_path)
        
        # Load queries (qid -> query text)
        self.queries = self._load_queries(queries_path)
        
        # Load qrels (qid -> {pid: relevance})
        self.qrels = self._load_qrels(qrels_path)
        
        # Load retrieval results (qid -> list of (pid, score))
        self.retrieval_results = self._load_retrieval_results(retrieval_results_path)
        
        # Create list of query IDs that have retrieval results
        self.query_ids = sorted(self.retrieval_results.keys())
        self.max_passages = max_passages
    
    def _load_collection(self, path: str) -> Dict[str, str]:
        """Load MS MARCO collection TSV"""
        collection = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pid, passage = parts[0], '\t'.join(parts[1:])
                    collection[pid] = passage
        return collection
    
    def _load_queries(self, path: str) -> Dict[str, str]:
        """Load TREC queries TSV"""
        queries = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    qid, query = parts[0], parts[1]
                    queries[qid] = query
        return queries
    
    def _load_qrels(self, path: str) -> Dict[str, Dict[str, int]]:
        """Load TREC qrels"""
        qrels = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, pid, relevance = parts[0], parts[1], parts[2], int(parts[3])
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][pid] = relevance
        return qrels
    
    def _load_retrieval_results(self, path: str) -> Dict[str, List[Tuple[str, float]]]:
        """Load retrieval results (qid -> [(pid, score), ...])"""
        results = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    qid = parts[0]
                    pid = parts[1]
                    score = float(parts[2]) if len(parts) > 2 else 0.0
                    if qid not in results:
                        results[qid] = []
                    results[qid].append((pid, score))
        return results
    
    def __len__(self):
        return len(self.query_ids)
    
    def load_example(self, idx: int) -> RankingExample:
        qid = self.query_ids[idx]
        query_text = self.queries.get(qid, '')
        
        # Get top passages from retrieval results
        retrieval_list = self.retrieval_results.get(qid, [])[:self.max_passages]
        
        # Create hits
        hits = []
        for rank, (pid, score) in enumerate(retrieval_list):
            passage_text = self.collection.get(pid, '')
            relevance = self.qrels.get(qid, {}).get(pid, 0)
            
            hits.append(Item(
                content=passage_text,
                id=pid,
                score=score,
                metadata={'rank': rank, 'relevance': relevance}
            ))
        
        # Create query
        query = Item(content=query_text, id=qid)
        
        # Get qrels for this query
        query_qrels = self.qrels.get(qid, {})
        
        return RankingExample(
            hits=hits,
            query=query,
            metadata={
                'query_id': qid,
                'qrels': query_qrels
            }
        )
```

#### 6.2 Evaluation Module Implementation
```python
# permsc/evaluation.py
import numpy as np
from typing import Dict, List

def dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute DCG@k"""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    relevances = np.array(relevances, dtype=float)
    gains = (2 ** relevances - 1) / np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(gains)

def ndcg_at_k(ranked_passage_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """Compute nDCG@k"""
    # Get relevance scores for ranked passages
    relevances = [qrels.get(pid, 0) for pid in ranked_passage_ids[:k]]
    
    # Compute DCG@k
    dcg = dcg_at_k(relevances, k)
    
    # Compute ideal DCG@k
    ideal_relevances = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_relevances, k)
    
    # Normalize
    if idcg == 0:
        return 0.0
    return dcg / idcg

def mrr_at_k(ranked_passage_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """Compute MRR@k"""
    for rank, pid in enumerate(ranked_passage_ids[:k], 1):
        if pid in qrels and qrels[pid] > 0:
            return 1.0 / rank
    return 0.0
```

---

### Step 7: Testing and Validation

#### 7.1 Unit Tests
- Test `MSMarcoDataset` loading with sample data
- Test evaluation metrics with known inputs
- Test aggregation pipeline with passage reranking examples

#### 7.2 Integration Test
- Run notebook on small subset (10-20 queries)
- Verify results match expected format
- Compare with baseline retrieval

---

### Step 8: Documentation

#### 8.1 Update README
- Add section on passage reranking experiment
- Document dataset requirements
- Add instructions for running the experiment

#### 8.2 Code Documentation
- Add docstrings to new classes and functions
- Document data format requirements
- Add examples in docstrings

---

## Key Differences from Sorting Tasks

1. **Evaluation:** Sorting tasks use Kendall tau correlation with ground truth order. Passage reranking uses nDCG@10 and MRR@10 with relevance judgments.

2. **Ground Truth:** Sorting tasks have a single correct order. Passage reranking has graded relevance (0, 1, 2, etc.) from qrels.

3. **Dataset Size:** Sorting tasks are small (50-100 examples). Passage reranking uses hundreds of queries with 100 passages each.

4. **Permutation Count:** Paper uses 20 permutations for passage reranking (vs 5 for sorting tasks in the notebook).

5. **Query-Passage Structure:** Passage reranking requires query-passage pairs, while sorting tasks only have items to sort.

---

## Dependencies to Add

- `pyserini` (optional, for first-stage retrieval)
- Standard libraries should be sufficient for evaluation metrics

---

## File Structure After Implementation

```
permsc/
  ├── data.py (add MSMarcoDataset)
  ├── evaluation.py (new file)
  └── ...

notebooks/
  └── passage-reranking.ipynb (new file)

data/
  ├── msmarco/
  │   └── collection.tsv
  ├── trec-dl19/
  │   ├── queries.tsv
  │   ├── qrels-pass.txt
  │   └── bm25-top100.tsv
  └── trec-dl20/
      ├── queries.tsv
      ├── qrels-pass.txt
      └── bm25-top100.tsv
```

---

## Next Steps Summary

1. **Immediate:** Download MS MARCO and TREC datasets
2. **Code:** Implement `MSMarcoDataset` class
3. **Code:** Implement evaluation metrics module
4. **Code:** Create passage reranking notebook
5. **Testing:** Run on small subset to validate
6. **Documentation:** Update README and add docstrings

