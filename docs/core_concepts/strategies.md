# Chunking Strategies

AutoChunks includes a library of 7+ primary chunking strategies, ranging from traditional mechanical splitters to advanced semantic engines.

## 1. Fixed-Length Chunker (Baseline)
The atomic baseline. It splits text into chunks of a strictly defined token count with an adaptive sliding overlap. While simple, it serves as the control group for measuring the semantic lift provided by more complex strategies.

*   **Logic**: Tokenization-based slicing with deterministic window stepping.
*   **Parameters**: `base_token_size` (int), `overlap` (int).

## 2. Recursive Character Chunker
A hierarchical "search-down" strategy. It attempts to maintain structural integrity by respecting a priority list of separators.

*   **Logic**: Recursive split → check size → split again if > limit. 
*   **Default Priority**: Double Newline (Paragraph) > Newline (Line) > Space (Word) > Character.
*   **Parameters**: `base_token_size`, `separators` (list).

## 3. Sentence-Aware Chunker
Ensures that logical units of thought (sentences) are never bifurcated.

*   **Logic**: Uses boundary detection to group sentences into the largest possible contiguous blocks that fit within the token budget.
*   **Use Case**: High-precision RAG where semantic fragmentation of a single sentence leads to categorical retrieval failures.

## 4. Semantic Chunker (Local Gradient)
Detects topic shifts by analyzing the semantic derivative across a sliding window of localized sentence embeddings.

*   **Logic**: 
    1.  Encode all sentences into embeddings.
    2.  Calculate a sliding window mean similarity.
    3.  Identify "Topic Chasms" where the similarity falls below a dynamic percentile-based threshold.
*   **Use Case**: Unstructured transcripts, long-form narratives, and fluid discussions.

## 5. Hybrid Semantic-Statistical Chunker
A sophisticated strategy that balances semantic topic shifting with token pressure constraints.

*   **Logic**: Boundary score calculation using a weighted scalar of semantic distance and current chunk length relative to target.
*   **Advantage**: Prevents "Stray Sentences" (semantic orphans) and "Runaway Chunks" (where topics never shift enough but size becomes unmanageable).

## 6. Layout-Aware Chunker (High-Fidelity)
Uses document structure as the primary boundary signal.

*   **Logic**: Parses the Markdown/HTML AST to identify H1-H3 headers, table starts, and blockquote boundaries. It weights these structural breaks higher than semantic similarity.
*   **Use Case**: Technical documentation, legal filings, and financial reports where the structure defines logical scope.

## 7. Parent-Child Chunker (Retrieval Strategy)
This is a retrieval strategy that indexes small child chunks for precise semantic search but retrieves a larger parent block to provide the LLM with sufficient context.

*   **Logic**: Bi-directional mapping between dense child vectors and sparse/semantic parent blocks.
*   **Advantage**: Solves the "Narrow Context" problem where a small chunk contains the answer but lacks the necessary logical surrounding to be useful for the LLM.

---

## Strategy Selection Matrix

| Strategy | Performance Overhead | Best For | Technical Insight |
| :--- | :--- | :--- | :--- |
| **Fixed** | Minimal | Homogeneous Text | Deterministic slicing. |
| **Semantic** | High (GPU) | Topic-Drifting Text | Sliding window cosine gradient. |
| **Hybrid** | Medium | Noisy Production Data | Weighted scalar of topic vs. size. |
| **Layout-Aware** | Low | Structured PDF/MD | AST/Structural boundary analysis. |
| **Recursive** | Minimal | Code/Markdown | Nested hierarchy splitting. |

---

**Next Steps**
*   [**Synthetic Ground Truth**](ground_truth.md)
*   [**Optimization Goals**](objectives.md)
*   [**Evaluation Engine**](evaluation.md)
