#!/usr/bin/env python3
"""
Generate BM25 retrieval results using Pyserini for TREC DL19/DL20 queries.

This script:
1. Indexes the MS MARCO collection using Pyserini
2. Retrieves top-100 passages for each query in TREC DL19/DL20
3. Saves results in the format expected by MSMarcoDataset
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from pyserini.search import SimpleSearcher
    from pyserini.index import IndexReader
except ImportError:
    print("Error: pyserini not installed. Run: pip install pyserini")
    sys.exit(1)


def index_collection(collection_path: str, index_path: str):
    """
    Index MS MARCO collection using Pyserini.
    
    Args:
        collection_path: Path to collection.tsv
        index_path: Path where index will be created
    """
    print(f"Indexing MS MARCO collection from {collection_path}...")
    print(f"Index will be created at {index_path}")
    
    # Convert collection to JSON format for Pyserini
    json_path = collection_path.replace('.tsv', '_pyserini.jsonl')
    print(f"Converting collection to JSON format: {json_path}")
    
    with open(collection_path, 'r', encoding='utf-8') as f_in, \
         open(json_path, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            parts = line.strip().split('\t', 1)
            if len(parts) >= 2:
                pid, passage = parts[0], parts[1]
                doc = {
                    'id': pid,
                    'contents': passage
                }
                f_out.write(json.dumps(doc) + '\n')
            
            if line_num % 100000 == 0:
                print(f"  Processed {line_num:,} passages...")
    
    print(f"✓ Converted {line_num:,} passages to JSON")
    
    # Index using Pyserini command line
    print(f"\nIndexing with Pyserini...")
    cmd = [
        'python', '-m', 'pyserini.index',
        '-collection', 'JsonCollection',
        '-generator', 'DefaultLuceneDocumentGenerator',
        '-threads', '9',
        '-input', json_path,
        '-index', index_path,
        '-storePositions', '-storeDocvectors', '-storeRaw'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error indexing collection:")
        print(result.stderr)
        sys.exit(1)
    
    print(f"✓ Index created at {index_path}")
    return index_path


def retrieve_top100(queries_path: str, index_path: str, output_path: str, k: int = 100):
    """
    Retrieve top-k passages for queries using BM25.
    
    Args:
        queries_path: Path to queries.tsv (qid, query)
        index_path: Path to Pyserini index
        output_path: Path to save results (TSV format: qid, pid, rank, score)
        k: Number of top passages to retrieve
    """
    print(f"\nRetrieving top-{k} passages for queries from {queries_path}...")
    
    # Load queries
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) >= 2:
                qid, query = parts[0], parts[1]
                queries[qid] = query
    
    print(f"Loaded {len(queries)} queries")
    
    # Initialize searcher
    searcher = SimpleSearcher(index_path)
    searcher.set_bm25()
    
    # Retrieve and save results
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for qid, query in queries.items():
            hits = searcher.search(query, k=k)
            
            for rank, hit in enumerate(hits, 1):
                pid = hit.docid
                score = hit.score
                # Format: qid, pid, rank, score
                f_out.write(f"{qid}\t{pid}\t{rank}\t{score}\n")
            
            if int(qid) % 50 == 0:
                print(f"  Processed query {qid}...")
    
    print(f"✓ Saved retrieval results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate BM25 retrieval results using Pyserini')
    parser.add_argument('--collection', type=str, required=True,
                       help='Path to MS MARCO collection.tsv')
    parser.add_argument('--queries', type=str, required=True,
                       help='Path to TREC queries.tsv')
    parser.add_argument('--index', type=str, default='indexes/msmarco-passage',
                       help='Path to Pyserini index (default: indexes/msmarco-passage)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for retrieval results (TSV format)')
    parser.add_argument('--k', type=int, default=100,
                       help='Number of top passages to retrieve (default: 100)')
    parser.add_argument('--skip-index', action='store_true',
                       help='Skip indexing (use existing index)')
    
    args = parser.parse_args()
    
    # Create index if needed
    if not args.skip_index:
        os.makedirs(os.path.dirname(args.index), exist_ok=True)
        index_collection(args.collection, args.index)
    else:
        print(f"Using existing index at {args.index}")
    
    # Retrieve
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    retrieve_top100(args.queries, args.index, args.output, k=args.k)
    
    print("\n✓ Retrieval complete!")


if __name__ == '__main__':
    main()

