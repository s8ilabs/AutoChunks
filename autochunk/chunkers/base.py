
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    meta: Dict[str, Any]

class BaseChunker:
    name = "base"
    def chunk(self, doc_id: str, text: str, **params) -> List[Chunk]:
        raise NotImplementedError
