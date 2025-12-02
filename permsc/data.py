__all__ = ['Item', 'RankingExample', 'Message', 'RankingDataset', 'MathSortDataset', 'GSM8KSortDataset',
           'WordSortDataset', 'CountrySortDataset', 'WordsDataset', 'MSMarcoDataset']

from copy import deepcopy
import json
from pathlib import Path
from typing import List, Any, Dict, Iterable

import nltk
import numpy as np
import pandas as pd
from pydantic import BaseModel


class Item(BaseModel):
    content: str
    score: float = 0
    id: str = ''
    metadata: dict = {}


class RankingExample(BaseModel):
    hits: List[Item]
    query: Item = None
    metadata: dict = {}

    def __getitem__(self, key) -> 'RankingExample':
        if isinstance(key, int):
            key = slice(key, key + 1)

        assert isinstance(key, slice), 'RankingExample can only be sliced with int or slice'
        assert key.step is None or key.step == 1, 'Slicing with step is not supported'

        metadata = self.metadata.copy()
        metadata['orig_indices'] = key

        # Update current permutation
        new_perm = np.empty(len(self.hits[key]), dtype=int)
        sort_idx = np.argsort(metadata['current_permutation'][key])
        new_perm[sort_idx] = np.arange(key.stop - key.start)
        metadata['current_permutation'] = new_perm

        return RankingExample(hits=deepcopy(self.hits[key]), query=deepcopy(self.query), metadata=metadata)

    @property
    def current_permutation(self) -> np.ndarray:
        return self.metadata.get('current_permutation', np.arange(len(self.hits)))

    def to_pyserini_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query.content,
            'hits': [{'content': hit.content, 'score': hit.score, 'rank': hit.metadata.get('rank', 0)} for hit in self.hits]
        }

    @classmethod
    def from_pyserini_dict(cls, d: Dict[str, Any]) -> 'RankingExample':
        hits = [Item(content=h['content'], score=h['score'], metadata={'rank': h['rank']}) for h in d['hits']]
        query = Item(content=d['query'])

        return cls(hits=hits, query=query)

    def sort_by(self, key, standardize: bool = False, reverse: bool = False) -> np.ndarray:
        perm_mask = np.argsort([key(hit) for hit in self.hits])

        if reverse:
            perm_mask = perm_mask[::-1]

        return self.randomize_order(perm_mask=perm_mask, standardize=standardize)

    def randomize_order(self, standardize: bool = False, perm_mask: np.ndarray = None) -> np.ndarray:
        perm_mask = np.random.permutation(len(self.hits)) if perm_mask is None else perm_mask
        self.metadata['current_permutation'] = self.metadata.get('current_permutation', np.arange(len(self.hits)))[perm_mask]
        self.hits = np.array(self.hits, dtype=object)[perm_mask].tolist()
        perm_mask = self.metadata['current_permutation']

        if standardize and 'current_permutation' in self.metadata:
            self.metadata['current_permutation'] = np.arange(len(self.hits))

        return perm_mask

    def split(self, split_size: int) -> Iterable['RankingExample']:
        for i in range(0, len(self), split_size):
            try:
                yield self[i:i + split_size]
            except ValueError:
                break

    def restore_order(self):
        perm_mask = self.metadata.get('current_permutation', np.arange(len(self.hits)))
        self.hits = np.array(self.hits, dtype=object)[np.argsort(perm_mask)].tolist()
        del self.metadata['current_permutation']

    def permuted_preferences_to_original_order(self, preferences: np.ndarray) -> np.ndarray:
        """Converts preference arrays over the permuted items to preference arrays over the original order."""
        perm_mask = self.metadata.get('current_permutation', np.arange(len(self.hits)))
        pref_restore_map = dict(zip(range(len(perm_mask)), perm_mask))
        pref_restore_map[-1] = -1

        return np.array([pref_restore_map[x] for x in preferences])

    def merge(self, other_example: 'RankingExample'):
        if other_example.metadata.get('orig_indices'):  # perform non-appending merge
            a, b = other_example.metadata['orig_indices']
            self.hits = self.hits[:a] + other_example.hits + self.hits[b:]
        else:
            self.hits += other_example.hits

    def __len__(self) -> int:
        return len(self.hits)


class RankingDataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int | slice) -> RankingExample:
        converted = False

        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
            converted = True

        examples = [self.load_example(i) for i in range(idx.start or 0, idx.stop or len(self), idx.step or 1)]

        return examples[0] if converted else examples

    def load_example(self, idx: int) -> RankingExample:
        raise NotImplementedError

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class MathSortDataset(RankingDataset):
    def __init__(self, path: str):
        df = pd.read_csv(path, sep='\t', quoting=3, escapechar='\\')
        self.dfs = [df for _, df in df.groupby('group')]

    def __len__(self):
        return len(self.dfs)

    def load_example(self, idx: int) -> RankingExample:
        df = self.dfs[idx]
        exprs = []

        for _, row in df.iterrows():
            exprs.append((row['expression'], row['answer']))

        exprs.sort(key=lambda x: x[1])
        hits = [Item(content=expr, score=1 / (idx + 1)) for idx, (expr, _) in enumerate(exprs)]

        return RankingExample(hits=hits)


class GSM8KSortDataset(RankingDataset):
    def __init__(self, path: str):
        self.question_sents_list = []

        for line in Path(path).read_text().splitlines():
            data = json.loads(line)
            sentences = nltk.tokenize.sent_tokenize(data['question'])

            if len(sentences) < 5 or len(sentences) > 10:
                continue

            self.question_sents_list.append(sentences)

    def __len__(self):
        return len(self.question_sents_list)

    def load_example(self, idx: int) -> RankingExample:
        sentences = self.question_sents_list[idx]
        hits = [Item(content=sent, score=1 / (idx + 1)) for idx, sent in enumerate(sentences)]

        return RankingExample(hits=hits)


class WordSortDataset(RankingDataset):
    def __init__(self, path: str):
        df = pd.read_csv(path, sep='\t', quoting=3, escapechar='\\')
        df['word_samples'] = df.word_samples.apply(lambda x: json.loads(x))
        self.df = df

    def __len__(self):
        return len(self.df)

    def load_example(self, idx: int) -> RankingExample:
        words = sorted(self.df.iloc[idx].word_samples)
        hits = [Item(content=word, score=1 / (idx + 1)) for idx, word in enumerate(words)]

        return RankingExample(hits=hits)


class WordsDataset(RankingDataset):
    def __init__(self, path: str):
        df = pd.read_csv(path, sep='\t', quoting=3, escapechar='\\')
        self.df = df

    def __len__(self):
        return 1

    def load_example(self, idx: int) -> RankingExample:
        words = self.df.words.tolist()
        hits = [Item(content=word) for idx, word in enumerate(words)]

        return RankingExample(hits=hits)


class CountrySortDataset(RankingDataset):
    def __init__(self, path: str):
        df = pd.read_csv(path, sep='\t', quoting=3, escapechar='\\')
        self.dfs = [x[1] for x in df.groupby('key')]

    def __len__(self):
        return len(self.dfs)

    def load_example(self, idx: int) -> RankingExample:
        df = self.dfs[idx]
        hits = []
        keys = ['percentage', 'number', 'year', 'rate']

        value = json.loads(df['value'].iloc[0])
        rel_key = None

        for key in keys:
            if key in value:
                rel_key = key
                break

        assert rel_key is not None, f'No relevant key found in {value}'

        for _, row in df.iterrows():
            value = json.loads(row['value'])
            hits.append(Item(content=row['country'], score=value[rel_key], metadata=dict(prompt=row['prompt'])))

        return RankingExample(hits=sorted(hits, key=lambda x: x.score, reverse=True))


class MSMarcoDataset(RankingDataset):
    """
    Dataset loader for MS MARCO passage reranking with TREC DL19/DL20.
    
    Loads:
    - MS MARCO collection (passage ID -> passage text)
    - TREC queries (query ID -> query text)
    - TREC qrels (query ID -> {passage ID: relevance})
    - Top-1000 retrieval results (query ID -> list of (passage ID, query, passage))
    """
    def __init__(
        self,
        collection_path: str,
        queries_path: str,
        qrels_path: str,
        top1000_path: str,
        max_passages: int = 100,
        use_passage_from_top1000: bool = True
    ):
        """
        Args:
            collection_path: Path to MS MARCO collection.tsv (pid, passage)
            queries_path: Path to TREC queries.tsv (qid, query)
            qrels_path: Path to TREC qrels-pass.txt (qid Q0 pid relevance)
            top1000_path: Path to top-1000 retrieval results (qid, pid, query, passage)
            max_passages: Maximum number of passages to use per query (default: 100)
            use_passage_from_top1000: If True, use passage text from top1000 file; 
                                      if False, look up from collection
        """
        self.max_passages = max_passages
        self.use_passage_from_top1000 = use_passage_from_top1000
        
        # Load collection (pid -> passage text)
        self.collection = self._load_collection(collection_path)
        
        # Load queries (qid -> query text)
        self.queries = self._load_queries(queries_path)
        
        # Load qrels (qid -> {pid: relevance})
        self.qrels = self._load_qrels(qrels_path)
        
        # Load top-1000 retrieval results (qid -> list of (pid, query, passage, rank))
        self.retrieval_results = self._load_top1000(top1000_path)
        
        # Create list of query IDs that have retrieval results
        self.query_ids = sorted(self.retrieval_results.keys())
    
    def _load_collection(self, path: str) -> Dict[str, str]:
        """Load MS MARCO collection TSV: pid, passage"""
        collection = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) >= 2:
                    pid, passage = parts[0], parts[1]
                    collection[pid] = passage
        return collection
    
    def _load_queries(self, path: str) -> Dict[str, str]:
        """Load TREC queries TSV: qid, query"""
        queries = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) >= 2:
                    qid, query = parts[0], parts[1]
                    queries[qid] = query
        return queries
    
    def _load_qrels(self, path: str) -> Dict[str, Dict[str, int]]:
        """Load TREC qrels: qid Q0 pid relevance"""
        qrels = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, pid, relevance = parts[0], parts[1], parts[2], int(parts[3])
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][pid] = relevance
        return qrels
    
    def _load_top1000(self, path: str) -> Dict[str, List[tuple]]:
        """
        Load retrieval results. Supports two formats:
        1. TREC format: qid, pid, query, passage (4 columns)
        2. Pyserini format: qid, pid, rank, score (4 columns)
        Returns: {qid: [(pid, query, passage, rank), ...]}
        """
        results = {}
        current_qid = None
        rank = 0
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    qid, pid = parts[0], parts[1]
                    
                    # Initialize results for this query if needed
                    if qid not in results:
                        results[qid] = []
                    
                    # Detect format: if 3rd column is numeric, it's Pyserini format
                    try:
                        float(parts[2])
                        # Pyserini format: qid, pid, rank, score
                        rank = int(parts[2]) - 1  # Convert to 0-indexed
                        score = float(parts[3])
                        query = self.queries.get(qid, '')
                        passage = self.collection.get(pid, '')
                    except ValueError:
                        # TREC format: qid, pid, query, passage
                        query = parts[2]
                        passage = parts[3]
                        # Reset rank when query changes
                        if current_qid != qid:
                            current_qid = qid
                            rank = 0
                        else:
                            rank += 1
                    
                    results[qid].append((pid, query, passage, rank))
        
        return results
    
    def __len__(self):
        return len(self.query_ids)
    
    def load_example(self, idx: int) -> RankingExample:
        qid = self.query_ids[idx]
        query_text = self.queries.get(qid, '')
        
        # Get top passages from retrieval results (limit to max_passages)
        retrieval_list = self.retrieval_results.get(qid, [])[:self.max_passages]
        
        # Create hits
        hits = []
        for rank, (pid, query_from_file, passage_from_file, _) in enumerate(retrieval_list):
            # Use passage from top1000 file or look up from collection
            if self.use_passage_from_top1000:
                passage_text = passage_from_file
            else:
                passage_text = self.collection.get(pid, '')
            
            # Get relevance label from qrels
            relevance = self.qrels.get(qid, {}).get(pid, 0)
            
            hits.append(Item(
                content=passage_text,
                id=pid,
                score=1.0 / (rank + 1),  # Inverse rank as score
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


class Message(BaseModel):
    role: str
    content: str
