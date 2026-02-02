"""
AutoChunks Quality Layer

World-class quality assurance tools for chunk evaluation and optimization.
"""

from .scorer import ChunkQualityScorer, ChunkQualityReport
from .deduplicator import ChunkDeduplicator, DeduplicationResult
from .overlap_optimizer import OverlapOptimizer, OverlapOptimizationResult
from .post_processor import ChunkPostProcessor, apply_post_processing, NATIVE_CHUNKERS, BRIDGE_CHUNKERS

__all__ = [
    # Scorer
    'ChunkQualityScorer',
    'ChunkQualityReport',
    
    # Deduplicator
    'ChunkDeduplicator', 
    'DeduplicationResult',
    
    # Overlap Optimizer
    'OverlapOptimizer',
    'OverlapOptimizationResult',
    
    # Post-Processor Pipeline
    'ChunkPostProcessor',
    'apply_post_processing',
    'NATIVE_CHUNKERS',
    'BRIDGE_CHUNKERS'
]

