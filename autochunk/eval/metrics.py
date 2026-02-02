
from __future__ import annotations
from typing import List
import math

def dcg(rels: List[int]) -> float:
    return sum((2**r - 1) / math.log2(i+2) for i, r in enumerate(rels))

def ndcg_at_k(rels: List[int], k: int) -> float:
    rels_k = rels[:k]
    ideal = sorted(rels_k, reverse=True)
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(rels_k) / denom

def mrr_at_k(rels: List[int], k: int) -> float:
    for i, r in enumerate(rels[:k]):
        if r > 0:
            return 1.0 / (i+1)
    return 0.0

def recall_at_k(rels: List[int], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    hit = sum(1 for r in rels[:k] if r > 0)
    return hit / total_relevant
