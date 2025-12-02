#!/usr/bin/env python3
"""
Simple test script to verify MSMarcoDataset loading (without full dependencies).
Run this after setting up the virtual environment.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from permsc import MSMarcoDataset, ndcg_at_k, mrr_at_k
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure virtual environment is activated and requirements are installed.")
    sys.exit(1)

# Test dataset loading
print("\nTesting MSMarcoDataset loading...")
try:
    ds = MSMarcoDataset(
        collection_path='data/msmarco/collection.tsv',
        queries_path='data/trec-dl19/msmarco-test2019-queries.tsv',
        qrels_path='data/trec-dl19/2019qrels-pass.txt',
        top1000_path='data/trec-dl19/msmarco-passagetest2019-top1000.tsv',
        max_passages=100
    )
    print(f"✓ Dataset loaded: {len(ds)} queries")
    
    # Test loading first example
    example = ds[0]
    print(f"✓ First query loaded: {example.query.content[:60]}...")
    print(f"✓ Number of passages: {len(example.hits)}")
    print(f"✓ First passage ID: {example.hits[0].id}")
    print(f"✓ Query ID in metadata: {example.metadata.get('query_id')}")
    print(f"✓ Qrels loaded: {len(example.metadata.get('qrels', {}))} relevant passages")
    
except Exception as e:
    print(f"✗ Dataset loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test evaluation metrics
print("\nTesting evaluation metrics...")
try:
    # Test case: ranked list with some relevant items
    ranked_ids = ['p1', 'p2', 'p3', 'p4', 'p5']
    qrels = {'p1': 0, 'p2': 2, 'p3': 1, 'p4': 0, 'p5': 0}
    
    ndcg = ndcg_at_k(ranked_ids, qrels, k=10)
    mrr = mrr_at_k(ranked_ids, qrels, k=10)
    
    print(f"✓ nDCG@10 computed: {ndcg:.4f}")
    print(f"✓ MRR@10 computed: {mrr:.4f}")
    
    # Test with actual example
    example = ds[0]
    qrels_example = example.metadata.get('qrels', {})
    ranked_ids_example = [hit.id for hit in example.hits[:10]]
    
    ndcg_example = ndcg_at_k(ranked_ids_example, qrels_example, k=10)
    mrr_example = mrr_at_k(ranked_ids_example, qrels_example, k=10)
    
    print(f"✓ nDCG@10 on real example: {ndcg_example:.4f}")
    print(f"✓ MRR@10 on real example: {mrr_example:.4f}")
    
except Exception as e:
    print(f"✗ Evaluation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")

