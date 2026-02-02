# RAGAS Evaluation Guide

AutoChunks integrates with [RAGAS](https://github.com/explodinggradients/ragas) to provide **LLM-based evaluation metrics** for your chunking strategies. This is a pluggable, optional component that complements the default statistical metrics (nDCG, MRR, Recall).

## What is RAGAS?

RAGAS (Retrieval Augmented Generation Assessment) is a framework that uses LLMs as "judges" to evaluate the quality of RAG pipelines. Unlike statistical metrics that measure retrieval precision, RAGAS metrics assess whether the retrieved context is actually useful for answering questions.

## Metrics Provided

| Metric | Description |
|--------|-------------|
| **Context Precision** | Measures if the ground truth is ranked highly in the retrieved context. High precision = relevant chunks appear at the top. |
| **Context Recall** | Measures if all the relevant information needed to answer the question is present in the context. |

## Supported LLM Providers

AutoChunks RAGAS integration supports multiple LLM backends:

| Provider | Setup Required | Best For |
|----------|---------------|----------|
| **OpenAI** (default) | Set `OPENAI_API_KEY` env var **OR** provide in UI | Production, highest quality |
| **Ollama** | Run Ollama locally | Air-gapped/offline, cost-free |
| **HuggingFace** | Install `transformers` | Local GPU inference |

---

## Usage

### CLI

**Basic (uses OpenAI by default):**
```bash
python -m autochunk optimize --docs ./my_documents --analyze-ragas
```

**With Ollama:**
```bash
# Start Ollama first (in another terminal)
ollama serve

# Run with explicit Ollama config
python -m autochunk optimize --docs ./my_documents \
    --analyze-ragas \
    --ragas-llm-provider ollama \
    --ragas-llm-model llama3.2
```

**With OpenAI:**
```bash
export OPENAI_API_KEY=sk-your-key-here

python -m autochunk optimize --docs ./my_documents \
    --analyze-ragas \
    --ragas-llm-provider openai \
    --ragas-llm-model gpt-4o-mini
```

### Python API

```python
from autochunk import AutoChunker, RagasConfig

# Using Ollama
chunker = AutoChunker(
    ragas_config=RagasConfig(
        enabled=True,
        llm_provider="ollama",
        llm_model="llama3.2",
        sample_size=20  # Limit to 20 QA pairs to save time/cost
    )
)

plan, report = chunker.optimize(documents="./my_docs")

# RAGAS metrics will appear in report["selected"]["metrics"]
print(report["selected"]["metrics"].get("context_precision"))
print(report["selected"]["metrics"].get("context_recall"))
```

### Web Dashboard (UI)

The dashboard provides a visual interface to enable RAGAS:

1. Open the dashboard (`autochunks serve`)
2. In the **Intelligence** card, check **"Enable RAGAS Analysis"**
3. Select your LLM Provider (OpenAI, Ollama, HuggingFace)
4. For OpenAI, provide your **LLM API Key** in the input field that appears.
5. Optionally specify a model name
6. Click **Start Optimization Pipeline**

Results will display:
- **Winner Card**: Shows RAGAS metrics (Context Precision, Context Recall) in amber
- **Candidate List**: RAGAS-evaluated candidates show a "✦ RAGAS" badge
- **Metrics**: Context Precision shown alongside standard metrics

### Web Dashboard API

```json
POST /api/optimize
{
  "documents_path": "./my_documents",
  "analyze_ragas": true,
  "ragas_llm_provider": "ollama",
  "ragas_llm_model": "llama3.2"
}
```

---

## Configuration Options

### RagasConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `False` | Enable/disable RAGAS evaluation |
| `llm_provider` | str | `"openai"` | LLM backend: `openai`, `ollama`, `huggingface` |
| `llm_model` | str | `None` | Model name (provider-specific) |
| `sample_size` | int | `20` | Max QA pairs to evaluate (limits cost) |
| `metrics` | list | `["context_precision", "context_recall"]` | Metrics to compute |

### Model Examples by Provider

| Provider | Example Models |
|----------|---------------|
| OpenAI | `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo` |
| Ollama | `llama3.2`, `llama3.1`, `gemma2`, `mistral`, `phi3` |
| HuggingFace | `microsoft/Phi-3-mini-4k-instruct`, `meta-llama/Llama-2-7b-chat-hf` |

---

## Installation

### Core Dependencies
```bash
pip install ragas datasets
```

### Provider-Specific Dependencies

**OpenAI:**
```bash
pip install langchain-openai
```

**Ollama:**
```bash
pip install langchain-ollama

# Also install and run Ollama: https://ollama.ai
ollama pull llama3.2
ollama serve
```

**HuggingFace:**
```bash
pip install transformers langchain-community torch
```

---

## How It Works

1. **Independent Retrieval**: RAGAS evaluation runs its own retrieval phase using the same embedding function, ensuring isolation from the main evaluation.

2. **Dataset Preparation**: For each QA pair, we retrieve top-k chunks and format as:
   - `question`: The synthetic query
   - `ground_truth`: The expected answer span
   - `contexts`: List of retrieved chunk texts

3. **LLM Judgment**: The configured LLM evaluates each sample, scoring context precision and recall.

4. **Metric Aggregation**: Results are averaged across all samples and returned alongside statistical metrics.

---

## Performance Considerations

| Factor | Impact | Recommendation |
|--------|--------|----------------|
| **Sample Size** | More samples = higher cost/time | Start with `sample_size=20` |
| **LLM Choice** | GPT-4 > GPT-3.5 > Local models | Use `gpt-4o-mini` for balance |
| **Local Models** | Free but slower, may be less accurate | Use `llama3.2` or `phi3` |

### Typical Evaluation Times

| Setup | Time per Candidate |
|-------|-------------------|
| OpenAI (gpt-4o-mini) | ~30-60 seconds |
| Ollama (llama3.2) | ~2-5 minutes |
| HuggingFace (Phi-3) | ~5-10 minutes (GPU) |

---

## Troubleshooting

### "No LLM provider available"

**Cause**: Neither OpenAI key nor Ollama detected.

**Fix**:
```bash
# Option 1: Set OpenAI key
export OPENAI_API_KEY=sk-xxx

# Option 2: Start Ollama
ollama serve
```

### "Missing RAGAS dependencies"

**Fix**:
```bash
pip install ragas datasets langchain-openai
```

### "langchain-ollama not installed"

**Fix**:
```bash
pip install langchain-ollama
```

### RAGAS returns NaN or errors

**Cause**: Often due to async execution issues or model incompatibility.

**Fix**:
```python
import nest_asyncio
nest_asyncio.apply()
```

---

## Comparison: Statistical vs RAGAS Metrics

| Aspect | Statistical (nDCG, MRR) | RAGAS (Context Precision/Recall) |
|--------|------------------------|----------------------------------|
| **Speed** | Milliseconds | Seconds to minutes |
| **Cost** | Free | LLM API costs (or compute) |
| **What it measures** | Ranking quality | Semantic usefulness |
| **Best for** | Fast iteration | Final validation |

### Recommendation

1. **Development**: Use statistical metrics only (fast, free)
2. **Validation**: Enable RAGAS on final candidates
3. **Production**: Run RAGAS on production samples periodically

---

## See Also

- [Evaluation Engine](../core_concepts/evaluation.md) - Core statistical metrics
- [Getting Started](../getting_started.md) - Quick start guide
- [RAGAS Documentation](https://docs.ragas.io/) - Official RAGAS docs
