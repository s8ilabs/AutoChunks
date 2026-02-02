
from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
import numpy as np
from ..chunkers.base import Chunk
from ..utils.text import count_tokens, split_sentences
from ..utils.logger import logger
import time

@dataclass
class ChunkQualityReport:
    """Comprehensive quality report for a chunk."""
    chunk_id: str
    overall_score: float  # 0-1, higher is better
    
    # Individual dimension scores (0-1)
    coherence_score: float      # Internal semantic consistency
    completeness_score: float   # Self-containedness
    density_score: float        # Information density
    boundary_score: float       # Quality of start/end boundaries
    size_score: float           # Optimal size relative to target
    
    # Detailed metrics
    token_count: int
    sentence_count: int
    avg_sentence_length: float
    
    # Flags
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ChunkQualityScorer:
    """
    World-Class Chunk Quality Scoring System.
    
    Evaluates chunks across multiple quality dimensions to identify
    problematic chunks and guide optimization.
    
    QUALITY DIMENSIONS:
    1. Coherence: Internal semantic consistency (embedding similarity)
    2. Completeness: Self-containedness (no dangling references)
    3. Density: Information richness vs fluff
    4. Boundaries: Clean starts and ends (no mid-sentence cuts)
    5. Size: Optimal length for the target use case
    
    SCORING METHODOLOGY:
    - Each dimension is scored 0-1
    - Weighted combination for overall score
    - Issue detection with actionable recommendations
    """

    # Patterns indicating incomplete boundaries
    INCOMPLETE_START_PATTERNS = [
        r'^[a-z]',           # Starts with lowercase
        r'^(and|but|or|so|then|however|therefore|thus|hence)\b',  # Starts with conjunction
        r'^(this|that|these|those|it|they|he|she)\b',  # Starts with pronoun
        r'^\.',              # Starts with period
        r'^\,',              # Starts with comma
    ]
    
    INCOMPLETE_END_PATTERNS = [
        r'[,;:]\s*$',        # Ends with comma/semicolon
        r'\b(and|or|but|the|a|an)\s*$',  # Ends with article/conjunction
        r'[^.!?\"\')\]]\s*$',  # Doesn't end with terminal punctuation
    ]
    
    # Filler words that reduce density
    FILLER_WORDS = {
        'very', 'really', 'quite', 'rather', 'somewhat', 'basically',
        'actually', 'literally', 'just', 'simply', 'obviously', 'clearly',
        'in order to', 'due to the fact that', 'at this point in time'
    }

    def __init__(self,
                 embedding_fn: Callable[[List[str]], List[List[float]]] = None,
                 target_token_size: int = 512,
                 weights: Dict[str, float] = None):
        """
        Initialize the quality scorer.
        
        Args:
            embedding_fn: Function to generate embeddings for coherence scoring.
            target_token_size: Ideal chunk size for size scoring.
            weights: Custom weights for each dimension.
        """
        self.embedding_fn = embedding_fn
        self.target_token_size = target_token_size
        self.weights = weights or {
            'coherence': 0.25,
            'completeness': 0.20,
            'density': 0.15,
            'boundary': 0.20,
            'size': 0.20
        }

    def score_chunk(self, chunk: Chunk) -> ChunkQualityReport:
        """
        Generate comprehensive quality report for a single chunk.
        
        Args:
            chunk: The chunk to evaluate
        
        Returns:
            ChunkQualityReport with all metrics
        """
        text = chunk.text
        issues = []
        recommendations = []
        
        # Basic metrics
        token_count = count_tokens(text)
        sentences = split_sentences(text)
        sentence_count = len(sentences)
        avg_sentence_length = np.mean([count_tokens(s) for s in sentences]) if sentences else 0
        
        # 1. Coherence Score
        coherence_score = self._score_coherence(sentences)
        if coherence_score < 0.6:
            issues.append("Low internal coherence - chunk may contain unrelated content")
            recommendations.append("Consider splitting at topic boundaries")
        
        # 2. Completeness Score
        completeness_score = self._score_completeness(text)
        if completeness_score < 0.7:
            issues.append("Chunk may not be self-contained")
            recommendations.append("Resolve pronouns and add context")
        
        # 3. Density Score
        density_score = self._score_density(text, sentences)
        if density_score < 0.5:
            issues.append("Low information density")
            recommendations.append("Remove filler words and redundant phrases")
        
        # 4. Boundary Score
        boundary_score = self._score_boundaries(text)
        if boundary_score < 0.7:
            issues.append("Incomplete boundaries detected")
            recommendations.append("Adjust split points to sentence boundaries")
        
        # 5. Size Score
        size_score = self._score_size(token_count)
        if size_score < 0.6:
            if token_count < self.target_token_size * 0.3:
                issues.append("Chunk is too small")
                recommendations.append("Merge with adjacent chunks")
            else:
                issues.append("Chunk is too large")
                recommendations.append("Split into smaller chunks")
        
        # Calculate overall score
        overall_score = (
            self.weights['coherence'] * coherence_score +
            self.weights['completeness'] * completeness_score +
            self.weights['density'] * density_score +
            self.weights['boundary'] * boundary_score +
            self.weights['size'] * size_score
        )
        
        return ChunkQualityReport(
            chunk_id=chunk.id,
            overall_score=overall_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            density_score=density_score,
            boundary_score=boundary_score,
            size_score=size_score,
            token_count=token_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            issues=issues,
            recommendations=recommendations
        )

    def score_chunks(self, chunks: List[Chunk]) -> List[ChunkQualityReport]:
        """
        Score multiple chunks with optimized batch embedding and progress logging.
        """
        if not chunks:
            return []

        logger.info(f"QualityScorer: Scoring {len(chunks)} chunks...")
        start_time = time.time()

        # Optimization: Pre-collect and batch embed all sentences for coherence scoring
        all_sentences = []
        chunk_sentence_maps = []
        
        if self.embedding_fn:
            logger.info(f"QualityScorer: Splitting sentences for batch embedding...")
            for chunk in chunks:
                sentences = split_sentences(chunk.text)
                start_idx = len(all_sentences)
                all_sentences.extend(sentences)
                end_idx = len(all_sentences)
                chunk_sentence_maps.append((sentences, list(range(start_idx, end_idx))))
            
            logger.info(f"QualityScorer: Batch embedding {len(all_sentences)} sentences...")
            embed_start = time.time()
            all_embeddings = self.embedding_fn(all_sentences)
            logger.info(f"QualityScorer: Embedding finished in {time.time()-embed_start:.2f}s")
            
            reports = []
            for i, chunk in enumerate(chunks):
                if i > 0 and i % 50 == 0:
                    logger.info(f"QualityScorer: Scoring progress {i}/{len(chunks)}...")
                
                sentences, indices = chunk_sentence_maps[i]
                embeddings = [all_embeddings[idx] for idx in indices]
                # Pass pre-calculated embeddings to a specialized internal method
                reports.append(self._score_chunk_with_cached_embeddings(chunk, sentences, embeddings))
            
            logger.info(f"QualityScorer: Total scoring finished in {time.time()-start_time:.2f}s")
            return reports
        else:
            # Fallback to serial scoring if no embedding function
            logger.info(f"QualityScorer: Using non-embedding fallback scoring...")
            results = [self.score_chunk(chunk) for chunk in chunks]
            logger.info(f"QualityScorer: Fallback scoring finished in {time.time()-start_time:.2f}s")
            return results

    def _score_chunk_with_cached_embeddings(self, 
                                           chunk: Chunk, 
                                           sentences: List[str], 
                                           sentence_embeddings: List[List[float]]) -> ChunkQualityReport:
        """Internal version of score_chunk that uses pre-calculated embeddings."""
        text = chunk.text
        issues = []
        recommendations = []
        
        # Basic metrics
        token_count = count_tokens(text)
        sentence_count = len(sentences)
        avg_sentence_length = np.mean([count_tokens(s) for s in sentences]) if sentences else 0
        
        # 1. Coherence Score (using pre-calculated embeddings)
        if self.embedding_fn:
            coherence_score = float(self._score_coherence_cached(sentences, sentence_embeddings))
        else:
            coherence_score = float(self._lexical_coherence(sentences))

        if coherence_score < 0.6:
            issues.append("Low internal coherence")
            recommendations.append("Consider splitting at topic boundaries")
            
        completeness_score = float(self._score_completeness(text))
        density_score = float(self._score_density(text, sentences))
        boundary_score = float(self._score_boundaries(text))
        size_score = float(self._score_size(token_count))
        
        overall_score = float(
            self.weights['coherence'] * coherence_score +
            self.weights['completeness'] * completeness_score +
            self.weights['density'] * density_score +
            self.weights['boundary'] * boundary_score +
            self.weights['size'] * size_score
        )
        
        return ChunkQualityReport(
            chunk_id=chunk.id, overall_score=overall_score,
            coherence_score=coherence_score, completeness_score=completeness_score,
            density_score=density_score, boundary_score=boundary_score,
            size_score=size_score, token_count=token_count,
            sentence_count=sentence_count, avg_sentence_length=float(avg_sentence_length),
            issues=issues, recommendations=recommendations
        )

    def _score_coherence_cached(self, sentences: List[str], embeddings: List[List[float]]) -> float:
        """Score coherence using provided embeddings."""
        if len(sentences) <= 1: return 1.0
        if not embeddings: return self._lexical_coherence(sentences)
        
        emb_array = np.array(embeddings)
        similarities = []
        for i in range(len(emb_array) - 1):
            for j in range(i + 1, len(emb_array)):
                norm_i = np.linalg.norm(emb_array[i])
                norm_j = np.linalg.norm(emb_array[j])
                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(emb_array[i], emb_array[j]) / (norm_i * norm_j)
                    similarities.append(sim)
        return np.mean(similarities) if similarities else 0.5

    def get_summary_stats(self, reports: List[ChunkQualityReport]) -> Dict[str, Any]:
        """Get aggregate statistics from multiple reports with plain Python types for serialization."""
        if not reports:
            return {}
        
        scores = [r.overall_score for r in reports]
        return {
            'count': len(reports),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'below_threshold': int(sum(1 for s in scores if s < 0.6)),
            'dimension_means': {
                'coherence': float(np.mean([r.coherence_score for r in reports])),
                'completeness': float(np.mean([r.completeness_score for r in reports])),
                'density': float(np.mean([r.density_score for r in reports])),
                'boundary': float(np.mean([r.boundary_score for r in reports])),
                'size': float(np.mean([r.size_score for r in reports]))
            }
        }

    def _score_coherence(self, sentences: List[str]) -> float:
        """Score internal semantic consistency."""
        if len(sentences) <= 1:
            return 1.0  # Single sentence is coherent by definition
        
        if self.embedding_fn is None:
            # Fallback: use lexical overlap
            return self._lexical_coherence(sentences)
        
        try:
            embeddings = np.array(self.embedding_fn(sentences))
            n = len(embeddings)
            if n <= 1: return 1.0
            
            # Vectorized pairwise cosine similarity
            # Normalize embeddings first
            norms = np.linalg.norm(embeddings, axis=1)
            norms[norms < 1e-9] = 1.0
            norm_embeddings = embeddings / norms[:, np.newaxis]
            
            # Similarity matrix (N x N)
            sim_matrix = norm_embeddings @ norm_embeddings.T
            
            # Extract upper triangle (excluding diagonal) for pairwise mean
            tri_indices = np.triu_indices(n, k=1)
            pair_similarities = sim_matrix[tri_indices]
            
            return float(np.mean(pair_similarities)) if pair_similarities.size > 0 else 0.5
        except Exception as e:
            logger.debug(f"Vectorized coherence scoring failed: {e}")
            return self._lexical_coherence(sentences)

    def _lexical_coherence(self, sentences: List[str]) -> float:
        """Fallback coherence using word overlap."""
        if len(sentences) <= 1:
            return 1.0
        
        word_sets = [set(s.lower().split()) for s in sentences]
        overlaps = []
        
        for i in range(len(word_sets) - 1):
            intersection = word_sets[i] & word_sets[i + 1]
            union = word_sets[i] | word_sets[i + 1]
            if union:
                overlaps.append(len(intersection) / len(union))
        
        return np.mean(overlaps) if overlaps else 0.5

    def _score_completeness(self, text: str) -> float:
        """Score self-containedness."""
        import re
        
        score = 1.0
        penalties = []
        
        # Check for unresolved pronouns at start
        first_sentence = split_sentences(text)[0] if split_sentences(text) else text[:100]
        pronoun_pattern = r'\b(this|that|these|those|it|they|he|she|him|her|them)\b'
        if re.search(pronoun_pattern, first_sentence.lower()):
            penalties.append(0.15)
        
        # Check for references to external context
        external_refs = [
            r'\babove\b', r'\bbelow\b', r'\bpreviously\b', r'\bfollowing\b',
            r'\bas mentioned\b', r'\bas discussed\b', r'\bsee \w+\b'
        ]
        for pattern in external_refs:
            if re.search(pattern, text.lower()):
                penalties.append(0.1)
        
        # Check for incomplete lists/enumerations
        if re.search(r'\b(firstly|first|1\.)\b', text.lower()):
            if not re.search(r'\b(secondly|second|2\.)\b', text.lower()):
                penalties.append(0.1)  # Started enumeration but incomplete
        
        return max(0.0, score - sum(penalties))

    def _score_density(self, text: str, sentences: List[str]) -> float:
        """Score information density."""
        if not text:
            return 0.0
        
        words = text.lower().split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Count filler words
        filler_count = sum(1 for w in words if w in self.FILLER_WORDS)
        filler_ratio = filler_count / word_count
        
        # Check for repetition
        unique_words = set(words)
        uniqueness_ratio = len(unique_words) / word_count
        
        # Sentence length variance (too uniform = formulaic)
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            length_variance = np.std(lengths) / (np.mean(lengths) + 1)
            variance_score = min(1.0, length_variance * 2)  # Some variance is good
        else:
            variance_score = 0.5
        
        density = (
            0.4 * (1 - filler_ratio * 5) +  # Penalize fillers
            0.4 * uniqueness_ratio +
            0.2 * variance_score
        )
        
        return max(0.0, min(1.0, density))

    def _score_boundaries(self, text: str) -> float:
        """Score quality of chunk boundaries."""
        import re
        
        score = 1.0
        
        # Check start
        for pattern in self.INCOMPLETE_START_PATTERNS:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                score -= 0.15
                break
        
        # Check end
        for pattern in self.INCOMPLETE_END_PATTERNS:
            if re.search(pattern, text.strip()):
                score -= 0.15
                break
        
        # Bonus for clean sentence boundaries
        text = text.strip()
        if text and text[-1] in '.!?"\'':
            score += 0.1
        
        return max(0.0, min(1.0, score))

    def _score_size(self, token_count: int) -> float:
        """Score chunk size relative to target."""
        if self.target_token_size == 0:
            return 1.0
        
        ratio = token_count / self.target_token_size
        
        # Optimal range: 0.7 - 1.3 of target
        if 0.7 <= ratio <= 1.3:
            return 1.0
        elif 0.5 <= ratio <= 1.5:
            return 0.8
        elif 0.3 <= ratio <= 2.0:
            return 0.5
        else:
            return 0.2
