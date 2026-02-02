# Configuration Reference

AutoChunks is designed with a hierarchical configuration system. While the dashboard handles the majority of use cases, the Python API and CLI allow for deep customization of the optimization engine, safety constraints, and network topology.

## AutoChunkConfig

The master configuration object that governs the entire optimization run.

```python
@dataclass
class AutoChunkConfig:
    mode: str = "light"                    # light | full
    embedding_provider: str = "hashing"    # hashing | local | openai | ollama
    embedding_model_or_path: str = "..."   # Model ID, Path, or API name
    embedding_api_key: str = None          # Optional key for cloud providers
    
    # Sub-Configurations
    eval_config: EvalConfig
    ragas_config: RagasConfig
    proxy_config: ProxyConfig
    safety: SafetyConstraints
    parallel: ParallelConfig
    network: NetworkConfig
    cache_dir: str = ".ac_cache"
```

### Intelligence and Embedding Settings

*   **embedding_provider**: Defines the vectorization backend.
    *   `hashing`: Mock provider for rapid structural testing (zero cost).
    *   `local`: High-fidelity local embeddings using Sentence-Transformers (requires GPU for speed).
    *   `openai`: Production-grade cloud embeddings from OpenAI.
    *   `ollama`: Locally hosted API-based embeddings.
*   **embedding_api_key**: A secure string used for OpenAI or other cloud-based embedding providers. This is filtered from all logs and serialized reports.

---

## EvalConfig

Determines how the engine weights different performance metrics to select a winner.

*   **objective**: Maps to the high-level **Optimization Goal** selected in the UI. 
    *   `balanced`: High-performance baseline for general RAG.
    *   `quality`: Maximizes mathematical accuracy regardless of cost.
    *   `latency`: Prioritizes strategies that yield the smallest context for the LLM.
    *   `cost`: Focuses on secondary storage efficiency and chunk count reduction.
*   **metrics**: A list of retrieval benchmarks calculated for every candidate (e.g., `ndcg@10`, `mrr@10`, `recall@50`).
*   **k**: The retrieval depth used during simulation. Defaults to 10.

---

## RagasConfig

Configures the LLM-based semantic validation layer.

*   **enabled**: Activates the RAGAS evaluation phase for top candidates.
*   **llm_provider**: Selects the LLM judge backend (`openai`, `ollama`, or `huggingface`).
*   **llm_model**: Specify the judge model (e.g., `gpt-4o-mini` or `llama3.2`).
*   **sample_size**: Limits the number of ground-truth pairs used for RAGAS to control cost and time.
*   **api_key**: The LLM API key. Masked in the UI and protected via `SecretStr` on the backend.

---

## ProxyConfig

Controls document sampling to enable optimization over massive datasets without linear time increases.

*   **enabled**: When true, AutoChunks uses a representative sample of your documents instead of the entire corpus.
*   **proxy_percent**: The percentage of documents to include in the sample (default 10%).
*   **cluster_k**: The number of semantic clusters used to ensure the sample is diversely representative of the whole corpus.

---

## ParallelConfig

Governs resource allocation during the "Tournament" phase.

*   **embedding_concurrency**: Number of parallel workers for vectorizing chunks.
*   **retriever_concurrency**: Parallel workers for simulating queries across candidates.
*   **batch_size**: The internal batch size for processing texts through embedding models.

---

## NetworkConfig

Security and environment settings for enterprise environments.

*   **local_models_path**: Path to pre-downloaded model weights to enable air-gapped operations.
*   **trusted_orgs**: A whitelist of Hugging Face organizations (e.g., `BAAI`, `sentence-transformers`) that the system is permitted to download from.

---

## SafetyConstraints

Defines the "Safe Zone" for the optimizer to prevent degenerative strategies.

*   **max_chunks_per_doc**: Hard limit to prevent runaway recursive splitting.
*   **min_avg_chunk_tokens**: Prevents the engine from selecting extremely small, noisy chunks.
*   **max_redundant_overlap_ratio**: Penalizes strategies with excessive overlap that would bloat index costs.

---

**See Also**
*   [Optimization Goals](../core_concepts/objectives.md)
*   [Evaluation Engine](../core_concepts/evaluation.md)
*   [Enterprise Security](enterprise_security.md)
