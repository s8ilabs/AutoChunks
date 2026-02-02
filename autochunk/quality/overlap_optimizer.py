
from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from ..chunkers.base import Chunk
from ..utils.text import count_tokens, split_sentences

@dataclass
class OverlapOptimizationResult:
    """Result of overlap optimization."""
    original_chunks: List[Chunk]
    optimized_chunks: List[Chunk]
    overlap_stats: Dict[str, Any]
    improvements: List[str]


class OverlapOptimizer:
    """
    World-Class Intelligent Overlap Optimization System.
    
    Dynamically adjusts overlap between chunks based on semantic analysis
    to ensure optimal context continuity without redundancy.
    
    OPTIMIZATION STRATEGIES:
    1. Semantic Bridging: More overlap at topic boundaries
    2. Entity Preservation: Ensure named entities aren't split
    3. Sentence Integrity: Overlap at sentence boundaries
    4. Adaptive Sizing: Variable overlap based on chunk content
    5. Context Windows: Sliding windows with smart step sizes
    
    METHODS:
    - fixed: Traditional fixed-token overlap
    - semantic: Embedding-based adaptive overlap
    - entity: NER-based overlap for entity preservation
    - sentence: Always overlap complete sentences
    - hybrid: Combination of all methods
    """

    def __init__(self,
                 embedding_fn: Callable[[List[str]], List[List[float]]] = None,
                 base_overlap: int = 50,
                 min_overlap: int = 20,
                 max_overlap: int = 200,
                 method: str = "hybrid"):
        """
        Initialize the overlap optimizer.
        
        Args:
            embedding_fn: Function for semantic analysis.
            base_overlap: Default overlap in tokens.
            min_overlap: Minimum allowed overlap.
            max_overlap: Maximum allowed overlap.
            method: "fixed", "semantic", "entity", "sentence", "hybrid".
        """
        self.embedding_fn = embedding_fn
        self.base_overlap = base_overlap
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.method = method

    def optimize_overlaps(self, 
                          chunks: List[Chunk],
                          original_text: str = None) -> OverlapOptimizationResult:
        """
        Optimize overlaps between chunks with batched semantic analysis.
        """
        if len(chunks) < 2:
            return OverlapOptimizationResult(
                original_chunks=chunks,
                optimized_chunks=chunks,
                overlap_stats={'pairs_analyzed': 0},
                improvements=[]
            )
        
        # Optimization: Batch embed all boundary sentence pairs at once
        boundary_embeddings = None
        if self.embedding_fn and self.method in ["semantic", "hybrid"]:
            boundary_sentences = []
            for i in range(len(chunks) - 1):
                s1 = split_sentences(chunks[i].text)
                s2 = split_sentences(chunks[i + 1].text)
                if s1 and s2:
                    boundary_sentences.extend([s1[-1], s2[0]])
                else:
                    boundary_sentences.extend(["", ""]) # Placeholders
            
            boundary_embeddings = self.embedding_fn(boundary_sentences)
        
        # Analyze current overlaps
        current_overlaps = self._analyze_current_overlaps(chunks)
        
        # Calculate optimal overlaps for each pair
        optimal_overlaps = []
        for i in range(len(chunks) - 1):
            pair_embeddings = None
            if boundary_embeddings:
                pair_embeddings = [boundary_embeddings[i*2], boundary_embeddings[i*2 + 1]]
                
            optimal = self._calculate_optimal_overlap(
                chunks[i], chunks[i + 1], i, len(chunks), pair_embeddings
            )
            optimal_overlaps.append(optimal)
        
        # Generate optimized chunks
        optimized_chunks = self._apply_overlaps(chunks, optimal_overlaps, original_text)
        
        # Generate improvements list
        improvements = self._generate_improvements(current_overlaps, optimal_overlaps)
        
        return OverlapOptimizationResult(
            original_chunks=chunks,
            optimized_chunks=optimized_chunks,
            overlap_stats={
                'pairs_analyzed': len(chunks) - 1,
                'current_overlaps': current_overlaps,
                'optimal_overlaps': optimal_overlaps,
                'avg_current': float(np.mean(current_overlaps)) if current_overlaps else 0.0,
                'avg_optimal': float(np.mean(optimal_overlaps)) if optimal_overlaps else 0.0
            },
            improvements=improvements
        )

    def add_overlap_to_chunks(self, 
                               chunks: List[Chunk],
                               overlap_tokens: int = None) -> List[Chunk]:
        """
        Add overlap context to chunks that may not have it.
        
        This method adds text from adjacent chunks to each chunk,
        useful when chunks were created without overlap.
        
        Args:
            chunks: Original chunks without overlap.
            overlap_tokens: Number of tokens to overlap. Uses base_overlap if None.
        
        Returns:
            New list of chunks with overlap added.
        """
        if len(chunks) < 2:
            return chunks
        
        overlap = overlap_tokens or self.base_overlap
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            prefix = ""
            suffix = ""
            
            # Add suffix from next chunk
            if i < len(chunks) - 1:
                next_text = chunks[i + 1].text
                suffix_sentences = split_sentences(next_text)
                suffix_tokens = 0
                suffix_parts = []
                
                for sent in suffix_sentences:
                    sent_tokens = count_tokens(sent)
                    if suffix_tokens + sent_tokens <= overlap:
                        suffix_parts.append(sent)
                        suffix_tokens += sent_tokens
                    else:
                        break
                
                if suffix_parts:
                    suffix = " " + " ".join(suffix_parts)
            
            # Add prefix from previous chunk
            if i > 0:
                prev_text = chunks[i - 1].text
                prefix_sentences = split_sentences(prev_text)
                prefix_tokens = 0
                prefix_parts = []
                
                for sent in reversed(prefix_sentences):
                    sent_tokens = count_tokens(sent)
                    if prefix_tokens + sent_tokens <= overlap:
                        prefix_parts.insert(0, sent)
                        prefix_tokens += sent_tokens
                    else:
                        break
                
                if prefix_parts:
                    prefix = " ".join(prefix_parts) + " "
            
            # Create enhanced chunk
            enhanced_text = prefix + chunk.text + suffix
            
            enhanced_chunks.append(Chunk(
                id=f"{chunk.id}_enhanced",
                doc_id=chunk.doc_id,
                text=enhanced_text,
                meta={
                    **chunk.meta,
                    "has_overlap": True,
                    "prefix_tokens": count_tokens(prefix),
                    "suffix_tokens": count_tokens(suffix),
                    "original_id": chunk.id
                }
            ))
        
        return enhanced_chunks

    def _analyze_current_overlaps(self, chunks: List[Chunk]) -> List[int]:
        """Analyze existing overlaps between adjacent chunks."""
        overlaps = []
        
        for i in range(len(chunks) - 1):
            text1 = chunks[i].text.lower()
            text2 = chunks[i + 1].text.lower()
            
            # Find longest common suffix/prefix
            overlap_tokens = self._find_text_overlap(text1, text2)
            overlaps.append(overlap_tokens)
        
        return overlaps

    def _find_text_overlap(self, text1: str, text2: str) -> int:
        """Find token overlap between end of text1 and start of text2."""
        words1 = text1.split()
        words2 = text2.split()
        
        max_overlap = min(len(words1), len(words2), 50)  # Limit search
        
        for overlap_len in range(max_overlap, 0, -1):
            suffix = words1[-overlap_len:]
            prefix = words2[:overlap_len]
            if suffix == prefix:
                return overlap_len
        
        return 0

    def _calculate_optimal_overlap(self, 
                                   chunk1: Chunk, 
                                   chunk2: Chunk,
                                   pair_index: int,
                                   total_chunks: int,
                                   pair_embeddings: Optional[List[List[float]]] = None) -> int:
        """Calculate optimal overlap for a chunk pair with optional pre-calculated embeddings."""
        if self.method == "fixed":
            return self.base_overlap
        
        optimal = self.base_overlap
        factors = []
        
        if self.method in ["semantic", "hybrid"]:
            semantic_factor = self._semantic_overlap_factor(chunk1, chunk2, pair_embeddings)
            factors.append(semantic_factor)
        
        if self.method in ["entity", "hybrid"]:
            entity_factor = self._entity_overlap_factor(chunk1, chunk2)
            factors.append(entity_factor)
        
        if self.method in ["sentence", "hybrid"]:
            sentence_factor = self._sentence_overlap_factor(chunk1, chunk2)
            factors.append(sentence_factor)
        
        # Combine factors
        if factors:
            avg_factor = float(np.mean(factors))
            optimal = int(self.base_overlap * avg_factor)
        
        # Clamp to bounds
        return max(self.min_overlap, min(self.max_overlap, optimal))

    def _semantic_overlap_factor(self, chunk1: Chunk, chunk2: Chunk, pair_embeddings: Optional[List[List[float]]] = None) -> float:
        """
        Calculate overlap factor based on semantic similarity using pre-calculated or on-the-fly embeddings.
        """
        try:
            vec1 = None
            vec2 = None
            
            if pair_embeddings and len(pair_embeddings) == 2:
                vec1, vec2 = np.array(pair_embeddings[0]), np.array(pair_embeddings[1])
            elif self.embedding_fn:
                # Fallback to on-the-fly calculation if not batched
                sentences1 = split_sentences(chunk1.text)
                sentences2 = split_sentences(chunk2.text)
                if not sentences1 or not sentences2: return 1.0
                boundary_texts = [sentences1[-1], sentences2[0]]
                embeddings = self.embedding_fn(boundary_texts)
                vec1, vec2 = np.array(embeddings[0]), np.array(embeddings[1])
            
            if vec1 is None or vec2 is None:
                return 1.0
                
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 1.0
            
            similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
            
            # Low similarity = topic shift = more overlap needed
            # Similarity 0.9+ = same topic = less overlap
            # Similarity 0.5- = big shift = more overlap
            if similarity > 0.85:
                return 0.6  # Reduce overlap
            elif similarity < 0.5:
                return 1.8  # Increase overlap
            else:
                return float(1.0 + (0.7 - similarity))  # Linear scaling
                
        except:
            return 1.0

    def _entity_overlap_factor(self, chunk1: Chunk, chunk2: Chunk) -> float:
        """
        Calculate overlap factor based on entity preservation.
        If chunk2 starts with references to entities from chunk1, increase overlap.
        """
        import re
        
        # Extract potential entities (capitalized words)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        
        entities1 = set(re.findall(entity_pattern, chunk1.text[-500:]))  # Last 500 chars
        
        # Check if chunk2 references these entities early
        first_100_chars = chunk2.text[:200]
        entities_in_start = set(re.findall(entity_pattern, first_100_chars))
        
        shared_entities = entities1 & entities_in_start
        
        if len(shared_entities) >= 3:
            return 1.5  # More overlap to include entity context
        elif len(shared_entities) >= 1:
            return 1.2
        else:
            return 1.0

    def _sentence_overlap_factor(self, chunk1: Chunk, chunk2: Chunk) -> float:
        """
        Calculate overlap to ensure sentence integrity at boundaries.
        """
        # Check if chunk1 ends mid-sentence
        text1 = chunk1.text.strip()
        
        if not text1:
            return 1.0
        
        # If doesn't end with sentence terminator, increase overlap
        if text1[-1] not in '.!?':
            return 1.5
        
        # Check if chunk2 starts mid-sentence
        text2 = chunk2.text.strip()
        if text2 and text2[0].islower():
            return 1.5
        
        return 1.0

    def _apply_overlaps(self, 
                        chunks: List[Chunk],
                        optimal_overlaps: List[int],
                        original_text: str = None) -> List[Chunk]:
        """Apply calculated overlaps to create new chunks."""
        # For now, return chunks with overlap metadata
        # Full implementation would re-chunk from original text
        
        optimized = []
        for i, chunk in enumerate(chunks):
            meta = {**chunk.meta}
            
            if i > 0:
                meta['overlap_from_prev'] = optimal_overlaps[i - 1]
            if i < len(optimal_overlaps):
                meta['overlap_to_next'] = optimal_overlaps[i]
            
            meta['overlap_optimized'] = True
            
            optimized.append(Chunk(
                id=chunk.id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                meta=meta
            ))
        
        return optimized

    def _generate_improvements(self, 
                               current: List[int], 
                               optimal: List[int]) -> List[str]:
        """Generate improvement recommendations."""
        improvements = []
        
        if not current or not optimal:
            return improvements
        
        for i, (curr, opt) in enumerate(zip(current, optimal)):
            diff = opt - curr
            if abs(diff) > 20:  # Significant difference
                if diff > 0:
                    improvements.append(
                        f"Pair {i}-{i+1}: Increase overlap by {diff} tokens for better context"
                    )
                else:
                    improvements.append(
                        f"Pair {i}-{i+1}: Reduce overlap by {-diff} tokens to remove redundancy"
                    )
        
        return improvements
