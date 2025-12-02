"""
Evaluation metrics for passage reranking tasks.
"""

__all__ = ['ndcg_at_k', 'mrr_at_k', 'dcg_at_k']

import numpy as np
from typing import List, Dict


def dcg_at_k(relevances: List[int], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at rank k.
    
    Args:
        relevances: List of relevance scores (0, 1, 2, etc.)
        k: Cutoff rank
        
    Returns:
        DCG@k score
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    
    relevances = np.array(relevances, dtype=float)
    # DCG = sum(rel_i / log2(i+1)) for i in [1, k]
    gains = (2 ** relevances - 1) / np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(gains)


def ndcg_at_k(ranked_passage_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank k.
    
    Args:
        ranked_passage_ids: List of passage IDs in ranked order
        qrels: Dictionary mapping passage ID to relevance score
        k: Cutoff rank (default: 10)
        
    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    # Get relevance scores for ranked passages
    relevances = [qrels.get(pid, 0) for pid in ranked_passage_ids[:k]]
    
    # Compute DCG@k
    dcg = dcg_at_k(relevances, k)
    
    # Compute ideal DCG@k (sorted by relevance, descending)
    ideal_relevances = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_relevances, k)
    
    # Normalize
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr_at_k(ranked_passage_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """
    Compute Mean Reciprocal Rank at rank k.
    
    Args:
        ranked_passage_ids: List of passage IDs in ranked order
        qrels: Dictionary mapping passage ID to relevance score (only positive scores matter)
        k: Cutoff rank (default: 10)
        
    Returns:
        MRR@k score (0.0 to 1.0)
    """
    for rank, pid in enumerate(ranked_passage_ids[:k], 1):
        if pid in qrels and qrels[pid] > 0:
            return 1.0 / rank
    return 0.0

