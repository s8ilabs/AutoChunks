# AutoChunks
### The Intelligent Data Optimization Layer for RAG Engineering

[![Version](https://img.shields.io/badge/version-0.08--alpha-blue)](https://github.com/s8ilabs/AutoChunks)
[![Documentation](https://img.shields.io/badge/docs-read--the--docs-teal)](https://autochunks.readthedocs.io/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

![AutoChunks Hero](docs/assets/hero_banner.png)

AutoChunks is a specialized engine designed to eliminate the guesswork from Retrieval-Augmented Generation (RAG). By treating chunking as an optimization problem rather than a set of heuristics, it empirically discovers the most performant data structures for your specific documents and retrieval models.

---

## From Heuristics to Evidence

Most RAG pipelines today rely on arbitrary settings—like a 512-token chunk size with a 10% overlap. These values are often chosen without validation, leading to:

*   **Fragmented Context**: Related information is split across multiple retrieval units.
*   **Semantic Noise**: Poorly defined boundaries dilute the signal-to-noise ratio in LLM prompts.
*   **Retrieval Gaps**: Critical information hidden in "dead zones" between chunks results in recall failure.

**AutoChunks replaces trial-and-error with a data-driven tournament.** It generates adversarial synthetic ground truth from your documents and pits over 15+ chunking strategies against each other to find the mathematical optimum for your corpus.

---

## Core Pillars

### The Vectorized Tournament
AutoChunks runs an exhaustive parallel search across multiple strategy families—Recursive, Semantic, Layout-Aware, and Hybrid. Every candidate is evaluated in a high-speed NumPy-accelerated retrieval simulation, measuring performance across hundreds of queries in seconds.

### Adversarial Synthetic QA
The system performs a structural audit of your documents to generate "needle-in-a-haystack" question-answer pairs. This ensures that your chunking strategy is optimized against real-world search intent, not just random text splits.

### Optimization Goals
Align your data engineering with your business objectives. Choose from intent-based presets that guide the engine toward specific outcomes:
*   **Balanced Ranking**: Optimizes for general-purpose retrieval quality.
*   **Speed and Precision**: Minimizes LLM reading time by prioritizing Rank #1 hits.
*   **Comprehensive Retrieval**: Prioritizes recall for compliance or legal use cases.
*   **Cost Efficiency**: Minimizes vector storage and inference costs for massive datasets.

---

## Advanced Feature Set

*   **Hybrid Semantic-Statistical Chunker**: Uses real-time embedding distance analysis to detect topic shifts while maintaining strict token limits.
*   **Framework Bridges**: Native adapters for LangChain, LlamaIndex, and Haystack, allowing you to benchmark and optimize your existing framework code directly.
*   **Layout-Aware Processing**: High-fidelity extraction that respects the nested structures of PDFs, HTML sections, and Markdown hierarchies.
*   **Fidelity Inspector**: A visual debugging dashboard to qualitatively verify how different strategies fragment complex documents.
*   **Enterprise Security**: Air-gap compatible. Supports local model deployment, SHA-256 binary fingerprinting for data privacy, and SecretStr protection for all cloud credentials.

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```
*Note: For GPU acceleration with Local Embeddings or Ragas, please refer to the [Getting Started guide](docs/getting_started.md).*

### Launch the Dashboard
The most effective way to optimize your data is through the visual interactive dashboard.
```bash
python -m autochunk.web.server
```
Navigate to `http://localhost:8000` to begin your first optimization run.

### Python API
```python
from autochunk import AutoChunker

# Initialize in Light Mode for rapid iteration
optimizer = AutoChunker(mode="light")

# Discover the optimal plan for your dataset
plan, report = optimizer.optimize(
    documents_path="./my_data_folder",
    objective="balanced"
)

# Apply the winning strategy
chunks = plan.apply("./new_documents", optimizer)
```

---

## Documentation and Resources

*   [Getting Started](docs/getting_started.md)
*   [The Optimization Lifecycle](docs/core_concepts/eval_flow.md)
*   [Metric Definitions and Scoring](docs/core_concepts/evaluation.md)
*   [RAGAS Semantic Evaluation](docs/guides/ragas_evaluation.md)

---

Developed for the RAG and LLM Community.
AutoChunks is released under the Apache License 2.0.
