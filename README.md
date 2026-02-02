# AutoChunks
### The Intelligent Data Optimization Layer for RAG Engineering

[![PyPI Version](https://img.shields.io/pypi/v/autochunks.svg)](https://pypi.org/project/autochunks/)
[![Documentation](https://img.shields.io/badge/docs-read--the--docs-teal)](https://autochunks.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

AutoChunks is a specialized engine designed to eliminate the guesswork from Retrieval-Augmented Generation (RAG). By treating chunking as an optimization problem rather than a set of heuristics, it empirically discovers the most performant data structures for your specific documents and retrieval models.

![AutoChunks Architecture](docs/assets/architecture.png)

---

## 🚀 Key Features

*   **The Vectorized Tournament**: Runs parallel searches across 15+ strategy families (Recursive, Semantic, Layout-Aware) using NumPy-accelerated simulation.
*   **Adversarial Synthetic QA**: Automatically generates "needle-in-a-haystack" QA pairs to test your data structure against real-world search intent.
*   **Multi-Objective Optimization**: Align engineering with goals like **Speed and Precision**, **Cost Efficiency**, or **Comprehensive Recall**.
*   **Framework Native**: Built-in bridges for LangChain, LlamaIndex, and Haystack.
*   **Enterprise Ready**: Air-gap compatible, local model support, and SHA-256 binary fingerprinting.

---

## 📦 Installation

Install the stable version from PyPI:
```bash
pip install autochunks
```

For GPU acceleration or RAGAS semantic evaluation, see the [Advanced Installation Guide](https://autochunks.readthedocs.io/en/latest/getting_started.html).

---

## 🛠️ Usage

### Launch the Dashboard
The easiest way to optimize is through the interactive visual dashboard:
```bash
autochunks serve
```
Navigate to `http://localhost:8000` to start your first optimization run.

### CLI Optimization
Search for the best plan directly from the terminal:
```bash
autochunks optimize --docs ./my_data_folder --mode light --objective balanced
```

### Python API
```python
from autochunk import AutoChunker

# Initialize and Discover the optimal plan
optimizer = AutoChunker(mode="light")
plan, report = optimizer.optimize(documents="./my_data", objective="balanced")

# Apply the winning strategy
chunks = plan.apply("./new_documents", optimizer)
```

---

## Development

If you want to contribute or build from source:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/s8ilabs/AutoChunks.git
   cd AutoChunks
   ```

2. **Setup virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # venv\Scripts\activate on Windows
   ```

3. **Install in editable mode**:
   ```bash
   pip install -e .
   ```

4. **Running Tests**:
   ```bash
   pytest tests/
   ```

---

## 📖 Documentation and Resources

*   [Full Documentation Portal](https://autochunks.readthedocs.io/en/latest/)
*   [PyPI Project Page](https://pypi.org/project/autochunks/)
*   [Getting Started Guide](docs/getting_started.md)
*   [The Optimization Lifecycle](docs/core_concepts/eval_flow.md)
*   [Metric Definitions and Scoring](docs/core_concepts/evaluation.md)

---

Developed with ❤️ for the RAG and LLM Community.
AutoChunks is released under the Apache License 2.0.
