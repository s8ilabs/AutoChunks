
from __future__ import annotations
from typing import List
from ...base import BaseChunker, Chunk

try:
    from langchain_text_splitters import HTMLSectionSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LangChainHTMLBridge(BaseChunker):
    name = "langchain_html"
    
    def chunk(self, doc_id: str, text: str, base_token_size: int = 512, overlap: int = 64, **params) -> List[Chunk]:
        if not LANGCHAIN_AVAILABLE:
            return []
            
        # HTMLSectionSplitter works differently, but we try to adapt
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ]
        try:
            from ....utils.logger import logger
            splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
            html_docs = splitter.split_text(text)
            
            # 2nd Pass: Enforce token size limits using RecursiveCharacterTextSplitter
            # This is the standard LangChain recipe for HTML RAG
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=base_token_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            
            final_docs = recursive_splitter.split_documents(html_docs)
            
            return [
                Chunk(
                    id=f"{doc_id}#lc_html#{i}",
                    doc_id=doc_id,
                    text=t.page_content,
                    # Merge metadata from both splitters
                    meta={**t.metadata, "chunk_index": i, "framework": "langchain", "strategy": "html_recursive"}
                ) for i, t in enumerate(final_docs)
            ]
        except Exception as e:
            from ....utils.logger import logger
            logger.error(f"LangChain HTML splitter failed: {e}")
            return []
