# Deploying to Production

AutoChunks is not just an analysis tool; it's a deployment pipeline. Once you have a `best_plan.yaml`, you can deploy it seamlessly into your existing RAG stack.

## The AutoChunks Adapter Pattern

We provide lightweight "Adapters" that wrap the AutoChunks logic into the native interfaces of popular frameworks.

We provide lightweight "Adapters" that wrap the AutoChunks logic into the native interfaces of popular frameworks.

---

## Deployment Logic
Once you have the `best_plan.yaml`, the integration follows a simple pattern: Load Plan → Initialize Adapter → Process Documents.

## 🦜 LangChain Integration

Use the `AutoChunkLangChainAdapter`. It behaves exactly like a standard LangChain `TextSplitter`.

```python
from autochunk.adapters.langchain import AutoChunkLangChainAdapter
from langchain.document_loaders import TextLoader

# 1. Load your optimized plan
splitter = AutoChunkLangChainAdapter(plan_path="best_plan.yaml")

# 2. Use it in your standard chain
loader = TextLoader("state_of_the_union.txt")
docs = loader.load()

# 3. Split
chunks = splitter.split_documents(docs)
print(f"Generated {len(chunks)} chunks using the optimized strategy.")
```

## 🦙 LlamaIndex Integration

Use the `AutoChunkLlamaIndexAdapter`. It functions as a LlamaIndex `NodeParser`.

```python
from autochunk.adapters.llamaindex import AutoChunkLlamaIndexAdapter
from llama_index import SimpleDirectoryReader, VectorStoreIndex

# 1. Initialize Adapter
node_parser = AutoChunkLlamaIndexAdapter(plan_path="best_plan.yaml")

# 2. Ingest
documents = SimpleDirectoryReader('data').load_data()
nodes = node_parser.get_nodes_from_documents(documents)

# 3. Build Index
index = VectorStoreIndex(nodes)
```

## 🌾 Haystack 2.0 Integration

Use the `AutoChunkHaystackAdapter`. It is a fully compliant Haystack `Component`.

```python
from autochunk.adapters.haystack import AutoChunkHaystackAdapter
from haystack import Pipeline

# 1. Initialize Component
chunker = AutoChunkHaystackAdapter(plan_path="best_plan.yaml")

# 2. Add to Pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("autochunker", chunker)
# ... add writer, embedder, etc ...

# 3. Run
indexing_pipeline.run({"autochunker": {"documents": docs}})
```

## JSON / REST API Deployment

If you are running a custom stack (e.g., Rust, Go, or pure Python), you can use the `apply` CLI command to pre-process documents before indexing.

```bash
# Process new files using the saved plan
autochunks apply --plan best_plan.yaml --docs ./incoming_data --out ./processed/chunks.json
```

Your ingestion pipeline can then simply read `chunks.json` and upload to Qdrant/Pinecone/Weaviate.

---

## Next Steps

*   [**Getting Started**](getting_started.md): Run your first optimization run.
*   [**The Optimization Lifecycle**](core_concepts/eval_flow.md): Understand how plans are generated.
*   [**API Reference**](api_reference.md): Technical details for the adapter classes.
