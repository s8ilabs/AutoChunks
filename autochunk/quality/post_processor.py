"""
AutoChunks Post-Processing Pipeline

Applies quality optimizations to chunks ONLY for native AutoChunks chunkers.
Bridge chunkers (LangChain, etc.) get raw output for fair comparison.
"""

from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional
from ..chunkers.base import Chunk
from ..utils.text import count_tokens, split_sentences
from ..utils.logger import logger
import time
from .scorer import ChunkQualityScorer
from .deduplicator import ChunkDeduplicator
from .overlap_optimizer import OverlapOptimizer

# Native AutoChunks chunkers that get post-processing
NATIVE_CHUNKERS = {
    "fixed_length",
    "recursive_character", 
    "sentence_aware",
    "semantic_local",
    "hybrid_semantic_stat",
    "parent_child",
    "layout_aware",
    "agentic",
    "proposition",
    "contextual_retrieval",
    "python_ast",
    "html_section"
}

# Bridge chunkers that get raw output (fair comparison)
BRIDGE_CHUNKERS = {
    "langchain_recursive",
    "langchain_character",
    "langchain_markdown",
    "langchain_token",
    "langchain_python",
    "langchain_html",
    "langchain_json"
}


class ChunkPostProcessor:
    """
    Post-processing pipeline for chunk optimization.
    
    Applied ONLY to native AutoChunks chunkers to ensure:
    1. Fair comparison with bridge chunkers (they get raw output)
    2. AutoChunks gets full pipeline benefits
    
    Pipeline Steps:
    1. Deduplication (optional) - Remove near-duplicate chunks
    2. Overlap Optimization (optional) - Add/adjust overlap for context
    3. Quality Scoring (always) - Add quality metrics to metadata
    """

    def __init__(self,
                 enable_dedup: bool = True,
                 enable_overlap_opt: bool = True,
                 embedding_fn: Callable[[List[str]], List[List[float]]] = None,
                 dedup_threshold: float = 0.90,
                 overlap_tokens: int = 50,
                 target_chunk_size: int = 512):
        """
        Initialize the post-processor.
        
        Args:
            enable_dedup: Whether to apply deduplication
            enable_overlap_opt: Whether to optimize overlaps
            embedding_fn: Embedding function for semantic operations
            dedup_threshold: Similarity threshold for deduplication (0.85-0.95)
            overlap_tokens: Target overlap in tokens
            target_chunk_size: Target chunk size for quality scoring
        """
        self.enable_dedup = enable_dedup
        self.enable_overlap_opt = enable_overlap_opt
        self.embedding_fn = embedding_fn
        self.dedup_threshold = dedup_threshold
        self.overlap_tokens = overlap_tokens
        self.target_chunk_size = target_chunk_size
        
        # Initialize components
        self.scorer = ChunkQualityScorer(
            embedding_fn=embedding_fn,
            target_token_size=target_chunk_size
        )
        
        self.deduper = ChunkDeduplicator(
            embedding_fn=embedding_fn,
            similarity_threshold=dedup_threshold,
            method="hybrid",
            strategy="keep_longest"  # Keep the most complete version
        )
        
        self.overlap_optimizer = OverlapOptimizer(
            embedding_fn=embedding_fn,
            base_overlap=overlap_tokens,
            min_overlap=20,
            max_overlap=100,
            method="hybrid"
        )

    def process(self, 
                chunks: List[Chunk], 
                chunker_name: str,
                return_metrics: bool = True) -> tuple[List[Chunk], Dict[str, Any]]:
        """
        Apply post-processing pipeline to chunks.
        
        Args:
            chunks: Input chunks from a chunker
            chunker_name: Name of the chunker that produced these chunks
            return_metrics: Whether to return quality metrics
        
        Returns:
            Tuple of (processed_chunks, quality_metrics)
        """
        metrics = {
            "post_processing_applied": False,
            "is_native_chunker": chunker_name in NATIVE_CHUNKERS,
            "original_count": len(chunks)
        }
        
        # Only apply optimizations to native chunkers
        if chunker_name not in NATIVE_CHUNKERS:
            # For bridges, just score quality but don't modify
            if return_metrics and chunks:
                quality_reports = self.scorer.score_chunks(chunks)
                metrics["avg_quality_score"] = sum(r.overall_score for r in quality_reports) / len(quality_reports)
                metrics["quality_dimensions"] = self.scorer.get_summary_stats(quality_reports).get("dimension_means", {})
            return chunks, metrics
        
        metrics["post_processing_applied"] = True
        processed_chunks = list(chunks)
        
        # Step 1: Deduplication
        if self.enable_dedup and len(processed_chunks) > 1:
            logger.info(f"[{chunker_name}] Post-processor: Starting deduplication (count={len(processed_chunks)})...")
            dp_start = time.time()
            dedup_result = self.deduper.deduplicate(processed_chunks)
            processed_chunks = dedup_result.kept_chunks
            metrics["dedup_removed"] = dedup_result.removed_count
            metrics["dedup_groups"] = len(dedup_result.duplicate_groups)
            logger.info(f"[{chunker_name}] Post-processor: Deduplication finished in {time.time()-dp_start:.2f}s (removed {dedup_result.removed_count})")
        
        # Step 2: Overlap Optimization
        if self.enable_overlap_opt and len(processed_chunks) > 1:
            logger.info(f"[{chunker_name}] Post-processor: Starting overlap optimization...")
            ov_start = time.time()
            # Add overlap context to chunks
            enhanced = self.overlap_optimizer.add_overlap_to_chunks(
                processed_chunks, 
                overlap_tokens=self.overlap_tokens
            )
            processed_chunks = enhanced
            metrics["overlap_enhanced"] = True
            logger.info(f"[{chunker_name}] Post-processor: Overlap optimization finished in {time.time()-ov_start:.2f}s")
        
        # Step 3: Quality Scoring (always applied, adds to metadata)
        if processed_chunks:
            logger.info(f"[{chunker_name}] Post-processor: Starting quality scoring (count={len(processed_chunks)})...")
            qs_start = time.time()
            quality_reports = self.scorer.score_chunks(processed_chunks)
            
            # Add quality scores to chunk metadata (Cast to float for serialization)
            for chunk, report in zip(processed_chunks, quality_reports):
                chunk.meta["quality_score"] = float(report.overall_score)
                chunk.meta["quality_coherence"] = float(report.coherence_score)
                chunk.meta["quality_completeness"] = float(report.completeness_score)
                chunk.meta["quality_density"] = float(report.density_score)
                chunk.meta["quality_boundary"] = float(report.boundary_score)
                chunk.meta["quality_size"] = float(report.size_score)
                if report.issues:
                    chunk.meta["quality_issues"] = report.issues
            
            # Aggregate metrics
            metrics["avg_quality_score"] = sum(r.overall_score for r in quality_reports) / len(quality_reports)
            metrics["quality_dimensions"] = self.scorer.get_summary_stats(quality_reports).get("dimension_means", {})
            metrics["chunks_with_issues"] = sum(1 for r in quality_reports if r.issues)
            logger.info(f"[{chunker_name}] Post-processor: Quality scoring finished in {time.time()-qs_start:.2f}s")
        
        metrics["final_count"] = len(processed_chunks)
        
        return processed_chunks, metrics


def apply_post_processing(chunks: List[Dict], 
                          chunker_name: str,
                          embedding_fn: Callable = None,
                          enable_dedup: bool = True,
                          enable_overlap: bool = True,
                          dedup_threshold: float = 0.90,
                          overlap_tokens: int = 50) -> tuple[List[Dict], Dict[str, Any]]:
    """
    Convenience function to apply post-processing to dict-format chunks.
    
    This is the main entry point for the autochunker.py integration.
    
    Args:
        chunks: List of chunk dictionaries with id, doc_id, text, meta
        chunker_name: Name of the chunker
        embedding_fn: Optional embedding function
        enable_dedup: Whether to deduplicate
        enable_overlap: Whether to optimize overlap
        dedup_threshold: Similarity threshold for deduplication
        overlap_tokens: Overlap size in tokens
    
    Returns:
        Tuple of (processed_chunk_dicts, quality_metrics)
    """
    # Convert dicts to Chunk objects
    chunk_objects = [
        Chunk(
            id=c["id"],
            doc_id=c["doc_id"],
            text=c["text"],
            meta=c.get("meta", {})
        ) for c in chunks
    ]
    
    # Create processor and run
    processor = ChunkPostProcessor(
        enable_dedup=enable_dedup,
        enable_overlap_opt=enable_overlap,
        embedding_fn=embedding_fn,
        dedup_threshold=dedup_threshold,
        overlap_tokens=overlap_tokens
    )
    
    processed_chunks, metrics = processor.process(chunk_objects, chunker_name)
    
    # Convert back to dicts
    result_dicts = [
        {
            "id": c.id,
            "doc_id": c.doc_id,
            "text": c.text,
            "meta": c.meta
        } for c in processed_chunks
    ]
    
    return result_dicts, metrics
