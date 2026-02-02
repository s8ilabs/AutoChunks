
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class EmbeddingFn:
    name: str
    dim: int
    fn: Callable[[List[str]], List[List[float]]]
    cost_per_1k_tokens: float = 0.0

    def __call__(self, texts: List[str]):
        return self.fn(texts)
