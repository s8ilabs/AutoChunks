
from __future__ import annotations
import json
from typing import List
from ...base import BaseChunker, Chunk

try:
    from langchain_text_splitters import RecursiveJsonSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LangChainJSONBridge(BaseChunker):
    name = "langchain_json"
    
    def chunk(self, doc_id: str, text: str, base_token_size: int = 512, **params) -> List[Chunk]:
        if not LANGCHAIN_AVAILABLE:
            return []
            
        try:
            if isinstance(text, str):
                data = json.loads(text)
            else:
                data = text
                
            splitter = RecursiveJsonSplitter(max_chunk_size=base_token_size)
            docs = splitter.split_text(data)
            
            return [
                Chunk(
                    id=f"{doc_id}#lc_json#{i}",
                    doc_id=doc_id,
                    text=t,
                    meta={"chunk_index": i, "framework": "langchain", "strategy": "json"}
                ) for i, t in enumerate(docs)
            ]
        except Exception as e:
            from ....utils.logger import logger
            logger.warning(f"LangChain JSON splitter skipped (input might not be JSON): {e}")
            return []
