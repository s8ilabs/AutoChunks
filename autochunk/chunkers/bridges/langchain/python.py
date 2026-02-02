
from __future__ import annotations
from typing import List
from ...base import BaseChunker, Chunk

try:
    from langchain_text_splitters import PythonCodeTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LangChainPythonBridge(BaseChunker):
    name = "langchain_python"
    
    def chunk(self, doc_id: str, text: str, base_token_size: int = 512, overlap: int = 64, **params) -> List[Chunk]:
        if not LANGCHAIN_AVAILABLE:
            return []
            
        from ....utils.text import count_tokens
        splitter = PythonCodeTextSplitter(
            chunk_size=base_token_size,
            chunk_overlap=overlap,
            length_function=count_tokens
        )
        
        docs = splitter.split_text(text)
        return [
            Chunk(
                id=f"{doc_id}#lc_py#{i}",
                doc_id=doc_id,
                text=t,
                meta={"chunk_index": i, "framework": "langchain", "strategy": "python"}
            ) for i, t in enumerate(docs)
        ]
