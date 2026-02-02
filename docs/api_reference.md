# API Reference

Detailed technical specification for the AutoChunks core engine.

---

## `AutoChunker`
*The primary controller for the optimization search and deployment lifecycle.*

```python
class AutoChunker:
    def __init__(
        self,
        mode: str = "light",
        embedding_provider: str = "hashing",
        embedding_model_or_path: str = "BAAI/bge-small-en-v1.5",
        eval_config: Optional[EvalConfig] = None,
        # ...
    ):
```

### Key Methods

#### `optimize()`
Runs the multi-objective search tournament across document configurations.

*   **Parameters:**
    *   `documents` (`str | List[Dict]`): Directory path or list of loaded document objects.
    *   `on_progress` (`Callable[[str, int], None]`): Callback for real-time status updates and telemetry.
    *   `sweep_params` (`Dict`): Overrides for hyperparameter ranges (Sizes, Overlaps).
*   **Returns:** `Tuple[Plan, Dict]` — The optimized `Plan` object and a comprehensive `Report` dictionary.

---

## `Plan`
*Represents a portable, serialized optimization result.*

### Attributes
*   **`generator_pipeline`**: The target chunking strategy name and validated hyperparameters.
*   **`metrics`**: The expected performance profile (nDCG, MRR, Recall) recorded during optimization.

### Methods
#### `apply(docs_path, chunker)`
Executes the strategy on a new corpus.
*   **Returns:** `List[Dict]` — A standard RAG-ready list of chunks with metadata.

---

## `EvalHarness`
*The vectorized evaluation and simulation engine.*

### Methods
#### `evaluate(chunks, qa)`
Performs the O(1) vectorized retrieval simulation.
*   **Performance Note**: Leverages BLAS-optimized matrix multiplication via NumPy for high-concurrency evaluation of large candidate sets.
