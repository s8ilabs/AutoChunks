# The Optimization Lifecycle

AutoChunks is more than a splitter; it is a closed-loop feedback system. It treats chunking as a scientific experiment where the "best" strategy is the one that mathematically maximizes retrieval success.

This page breaks down the technical lifecycle of an optimization run—the logic that powers the autonomous engine.

---

## Phase 1: Context Capture (QA Generation)

Before we can evaluate a strategy, we need a "Ground Truth." We create a high-fidelity benchmark (the Golden Set) specifically tailored to your documents. See the [**Synthetic Ground Truth**](ground_truth.md) guide for details.

1.  **Iterative Sampling**: Documents are analyzed and sampled to ensure the evaluation set is representative of the whole corpus.
2.  **Semantic Junction Identification**: The engine identifies high-entropy points in the text (e.g., transitions between topics) where a "bad" splitter is likely to fail.
3.  **Synthesis**:
    *   **Heuristic Mode**: Rapidly creates Question/Answer pairs based on sentence boundaries and layout markers.
    *   **LLM Mode**: (Optional) Uses a generator model to paraphrase sections into natural language questions.
4.  **Content-Based Caching**: QA pairs are keyed by the SHA-256 hash of the text content. If a file is renamed but the text is identical, AutoChunks reclaims the previous QA pairs instantly.

---

## Phase 2: The Tournament (Candidate Loop)

AutoChunks fans out and executes the following steps in parallel for every candidate configuration (e.g., "Semantic with 512 tokens" vs. "Recursive with 10% overlap").

### 1. Physical Chunking
The document text is passed to the candidate chunker (Native or Bridge). The output is a list of chunks with unique IDs and source metadata.

### 2. Post-Processing (Native Only)
For native AutoChunks splitters, we apply a "Clean Room" pass:
*   **Cosine Deduplication**: Pruning semantic redundancies within the document.
*   **Boundary Correction**: Ensuring chunks don't truncate words or mid-sentence logic.

### 3. Noise Injection and Indexing
To simulate a real-world vector database, we don't just index your 5 sample documents.

*   **Distractor Ingress**: We inject "Noise Chunks" (irrelevant text blocks) into the internal index to test the model's ability to differentiate signal from noise.
*   **Vectorization**: The EmbeddingProvider transforms chunks into vectors.
    *   **Local Backend**: Uses sentence-transformers for high-fidelity local simulation.
    *   **Cloud Backend**: Native support for OpenAI and Ollama APIs.
    *   **Hashing**: Used for structural testing and rapid iteration.
*   **Vectorized Loading**: Chunks are loaded into a NumPy-accelerated flat index for high-speed search simulation.

---

## Phase 3: Retrieval Simulation (The Trial)

This is where the benchmark happens.

1.  **Mass Vector Search**: All synthetic questions from Phase 1 are embedded and fired at the index in one massive batch operation using matrix duplication logic.
2.  **Top-K Retrieval**: For every question, the engine retrieves the top results (usually K=10) across the entire "Noisy" index.

---

## Phase 4: Scoring and Validation (The Jury)

We don't "guess" if the result is good; we verify it against the Golden Set code. For every retrieved Hit:

| Logic | Scoring Mechanism | Purpose |
| :--- | :--- | :--- |
| **Document Match** | Binary (`doc_id == target_id`) | Ensures the chunk isn't a "False Positive" from a different file. |
| **Precision Match** | Substring (`answer_span in chunk`) | Checks if the exact answer is present in the retrieved context. |
| **Token Coverage** | Jaccard Similarity (`tokens_a ∩ tokens_b`) | Identifies partial relevance if the answer was split across boundaries. |
| **Ranking Position** | Position Penalty | Penalizes the strategy if the perfect answer is at Rank #10 instead of Rank #1. |

---

## Phase 5: Goal Selection

Finally, the Multi-Objective Scorer combines the metrics into a single scalar value based on your selected [**Optimization Goal**](objectives.md):

$$ \text{Score} = (w_q \times Q) + (w_m \times M) + (w_c \times C) - \text{Penalty}_{\text{cost}} $$

The strategy with the highest final score is crowned the **Winner** and serialized as your `best_plan.yaml`. More details on result interpretation can be found in the [**Evaluation Engine**](evaluation.md).
