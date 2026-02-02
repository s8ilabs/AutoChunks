# Synthetic Ground Truth

Evaluating retrieval performance is difficult without labeled data—the "Gold Standard" where you know exactly which query should retrieve which specific context. AutoChunks solves this by generating a Synthetic Ground Truth dataset directly from your documents.

## The Generation Pipeline

AutoChunks follows a multi-stage process to create high-quality, retrieval-ready evaluation pairs.

### 1. Extraction and Sampling
AutoChunks identifies representative segments from your corpus using layout-aware analysis. If Proxy Sampling is enabled, it selects a statistically diverse subset of your documents to ensure the ground truth can be generated and evaluated in seconds rather than hours.

### 2. High-Fidelity Question Generation
A dedicated generator analyzes the selected segments to create specific questions.
*   **Context-Bound**: Questions are designed so they can only be answered by the specific target text span.
*   **Natural Language**: The generator mimics how real users ask questions, avoiding mechanical "keyword" queries.

### 3. Adversarial Paraphrasing
To ensure the retrievers are truly understanding the semantic intent and not just matching keywords, we apply Adversarial Paraphrasing. By reframing questions (e.g., changing "revenue" to "fiscal performance"), the engine forces the embedding models to rely on semantic topology rather than surface-level lexical overlap.

## Adversarial Synthesis Engine

Standard QA generation is often too easy for modern LLMs. AutoChunks uses an adversarial approach to specifically pressure-test the points where RAG systems usually fail: chunk boundaries.

### Semantic Junctions
We identify "Semantic Junctions"—points in a document where topics shift or sentences are loosely coupled. These are the locations where a naive chunker (like fixed-length) is most likely to fragment the context.

### Context-Targeting QA
We generate questions that deliberately require information from both sides of a potential break point. If a chunker splits the text at that junction, the retriever will fail to provide the full context to the LLM judge, causing the score to drop. This is how AutoChunks empirically detects and prevents "Context Leakage."

### Boundary Probing
The generator creates distractor-aware questions. It identifies parts of the document that are semantically similar to the target but factually irrelevant, testing the retriever's ability to distinguish between subtle contextual differences.

---

**Next Steps**: After the Ground Truth is generated, the engine enters the [**Tournament Phase**](eval_flow.md#phase-2-the-tournament-candidate-loop) to evaluate candidate strategies.
