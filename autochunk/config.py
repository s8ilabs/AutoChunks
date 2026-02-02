
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any

@dataclass
class EvalConfig:
    metrics: List[str] = field(default_factory=lambda: ["mrr@10", "ndcg@10", "recall@50"]) 
    objective: str = "balanced"  # quality|cost|latency|balanced
    k: int = 10
    latency_target_ms: int = 250
    cost_budget_usd: float = 5.0
    parent_child_eval: bool = False

@dataclass
class RetrievalStrategy:
    type: str = "standard"  # or parent_child
    child_token_size: int = 128
    parent_token_size: int = 1024

@dataclass
class ProxyConfig:
    enabled: bool = False
    cluster_k: int = 5
    proxy_percent: int = 10
    verify_percent: int = 20

@dataclass
class RagasConfig:
    enabled: bool = False
    metrics: List[str] = field(default_factory=lambda: ["context_precision", "context_recall"])
    sample_size: int = 20  # Limit RAGAS to a subset to save costs
    llm_provider: str = "openai"  # openai|ollama|huggingface
    llm_model: Optional[str] = None  # Model name (e.g., "gpt-4o-mini", "llama3.2", "microsoft/Phi-3-mini-4k-instruct")
    api_key: Optional[str] = field(default=None, repr=False) # API Key for OpenAI/Cloud providers

@dataclass
class SafetyConstraints:
    max_chunks_per_doc: int = 5000
    min_avg_chunk_tokens: int = 120
    max_redundant_overlap_ratio: float = 0.35

@dataclass
class ParallelConfig:
    embedding_concurrency: int = 4
    retriever_concurrency: int = 4
    batch_size: int = 32

@dataclass
class NetworkConfig:
    proxy_url: Optional[str] = None
    local_models_path: Optional[str] = None
    trusted_orgs: List[str] = field(default_factory=lambda: ["ds4sd", "RapidAI", "BAAI", "sentence-transformers"])

@dataclass
class TokenizerConfig:
    name: str = "whitespace"
    vocab_source: str = "custom"

@dataclass
class AutoChunkConfig:
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    retrieval_strategy: RetrievalStrategy = field(default_factory=RetrievalStrategy)
    proxy_config: ProxyConfig = field(default_factory=ProxyConfig)
    ragas_config: RagasConfig = field(default_factory=RagasConfig)
    safety: SafetyConstraints = field(default_factory=SafetyConstraints)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    embedding_provider: str = "hashing"  # hashing|local|tei|openai
    embedding_model_or_path: str = "BAAI/bge-small-en-v1.5"
    embedding_api_key: Optional[str] = field(default=None, repr=False) # API Key for cloud embedding providers
    mode: str = "light"  # light|full|incremental
    cache_dir: str = ".ac_cache"
    telemetry_enabled: bool = False  # Enable Tracing (Arize Phoenix)
    metadata_enrichment: Dict[str, Any] = field(default_factory=dict)
