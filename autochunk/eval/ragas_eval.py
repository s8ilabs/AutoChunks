from typing import List, Dict, Any, Optional
from ..config import RagasConfig
from ..utils.logger import logger
import os

class RagasEvaluator:
    """
    Pluggable RAGAS Evaluator that supports multiple LLM backends:
    - OpenAI (default, requires OPENAI_API_KEY)
    - Ollama (local, requires ollama running)
    - HuggingFace (local, requires GPU recommended)
    """
    
    def __init__(self, config: RagasConfig):
        self.config = config
    
    def _get_llm(self):
        """
        Returns the appropriate LLM wrapper based on config.
        Priority: config.llm_provider > OPENAI_API_KEY detection > fallback error
        """
        provider = getattr(self.config, 'llm_provider', 'auto')
        model_name = getattr(self.config, 'llm_model', None)
        
        # Auto-detect: Check for available providers
        if provider == 'auto':
            if os.environ.get("OPENAI_API_KEY"):
                provider = 'openai'
            else:
                # Try Ollama as fallback (common for local setups)
                try:
                    import requests
                    resp = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if resp.status_code == 200:
                        provider = 'ollama'
                        logger.info("RagasEvaluator: Auto-detected Ollama running locally")
                except:
                    pass
        
        if provider == 'openai':
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            api_key = getattr(self.config, 'api_key', None) or os.environ.get("OPENAI_API_KEY")
            llm = ChatOpenAI(model=model_name or "gpt-4o-mini", temperature=0, api_key=api_key)
            return LangchainLLMWrapper(llm)
        
        elif provider == 'ollama':
            try:
                from langchain_ollama import ChatOllama
                from ragas.llms import LangchainLLMWrapper
                llm = ChatOllama(model=model_name or "llama3.2", temperature=0)
                # Set dummy key to prevent RAGAS from complaining
                os.environ.setdefault("OPENAI_API_KEY", "sk-not-used-for-ollama")
                return LangchainLLMWrapper(llm)
            except ImportError:
                logger.warning("RagasEvaluator: langchain-ollama not installed. Install with: pip install langchain-ollama")
                return None
        
        elif provider == 'huggingface':
            try:
                from langchain_community.llms import HuggingFacePipeline
                from ragas.llms import LangchainLLMWrapper
                from transformers import pipeline
                
                pipe = pipeline("text-generation", model=model_name or "microsoft/Phi-3-mini-4k-instruct", max_new_tokens=512)
                llm = HuggingFacePipeline(pipeline=pipe)
                os.environ.setdefault("OPENAI_API_KEY", "sk-not-used-for-hf")
                return LangchainLLMWrapper(llm)
            except ImportError:
                logger.warning("RagasEvaluator: transformers/langchain-community not installed")
                return None
        
        return None
        
    def run(self, chunks: List[Dict], qa: List[Dict], embedding_fn=None, k: int = 5) -> Dict[str, Any]:
        """
        Runs RAGAS evaluation independently.
        Returns a dictionary of RAGAS-specific metrics (e.g., context_precision, context_recall).
        """
        if not self.config.enabled:
            return {}

        try:
            # Import lazily so core AutoChunks doesn't crash if RAGAS isn't installed
            from ragas import evaluate
            from ragas.metrics import context_precision, context_recall
            from datasets import Dataset
            
            # Use AutoChunks internal retrieval logic to support the 'contexts' creation
            from ..retrieval.in_memory import InMemoryIndex
        except ImportError as e:
            logger.warning(f"RagasEvaluator: Missing dependencies ({e}). Install with 'pip install ragas datasets'")
            return {"error": "Missing RAGAS dependencies"}

        logger.info("RagasEvaluator: Starting RAGAS evaluation...")
        
        # 0. Get the LLM
        llm = self._get_llm()
        if llm is None:
            logger.warning("RagasEvaluator: No LLM available. Set OPENAI_API_KEY or start Ollama locally.")
            return {"error": "No LLM provider available. Set OPENAI_API_KEY or run Ollama locally."}
        
        # 1. Perform Retrieval
        logger.info("RagasEvaluator: Performing independent retrieval step...")
        
        if embedding_fn is None:
            logger.warning("RagasEvaluator: No embedding_fn provided. Cannot perform retrieval.")
            return {}
        
        # Truncate texts to avoid embedding model max length errors
        # Most models have 512 token limit (~2000 chars is safe)
        MAX_CHARS = 1800
        def truncate(text: str) -> str:
            return text[:MAX_CHARS] if len(text) > MAX_CHARS else text
            
        chunk_texts = [truncate(c["text"]) for c in chunks]
        
        try:
            chunk_vectors = embedding_fn(chunk_texts)
        except RuntimeError as e:
            if "expanded size" in str(e) or "512" in str(e):
                logger.error(f"RagasEvaluator: Embedding model max length exceeded. Try smaller chunks.")
                return {"error": "Embedding model max length exceeded"}
            raise
        
        index = InMemoryIndex(dim=len(chunk_vectors[0]))
        index.add(chunk_vectors, chunks)
        
        ragas_data = {
            "question": [],
            "ground_truth": [],
            "contexts": [] 
        }
        
        valid_items_count = 0
        limit = self.config.sample_size if self.config.sample_size > 0 else len(qa)
        
        qa_subset = qa[:limit]
        query_texts = [truncate(q["query"]) for q in qa_subset]
        
        try:
            query_vectors = embedding_fn(query_texts)
        except RuntimeError as e:
            if "expanded size" in str(e):
                logger.error(f"RagasEvaluator: Query embedding failed - text too long")
                return {"error": "Query text too long for embedding model"}
            raise
            
        batch_hits = index.search(query_vectors, top_k=k)
        
        for item, hits in zip(qa_subset, batch_hits):
            contexts = [index.meta[idx]["text"] for idx, _ in hits]
            
            ragas_data["question"].append(item["query"])
            ragas_data["ground_truth"].append(item["answer_span"]) 
            ragas_data["contexts"].append(contexts)
            valid_items_count += 1

        if valid_items_count == 0:
            logger.warning("RagasEvaluator: No valid QA items found. Skipping.")
            return {}

        dataset = Dataset.from_dict(ragas_data)
        
        # 2. Run Evaluation with configured LLM
        logger.info(f"RagasEvaluator: Running evaluation with {valid_items_count} samples...")
        
        # Determine safe limit (characters)
        # 1 token ~= 4 chars. We allow 10% buffer.
        try:
            model_limit = 512 # Fallback
            
            # Check for Hashing
            is_hashing = getattr(embedding_fn, "name", "").startswith("hashing") or "HashingEmbedding" in str(type(embedding_fn))
            
            if is_hashing:
                 # Ragas might still need *some* limit for internal processing but Hashing isn't constrained
                 model_limit = 250_000
                 SAFE_CHAR_LIMIT = 1_000_000
            else:
                if hasattr(embedding_fn, "max_seq_length"):
                    model_limit = embedding_fn.max_seq_length
                elif hasattr(embedding_fn, "__self__") and hasattr(embedding_fn.__self__, "max_seq_length"):
                     # Handle bound methods like encoder.embed_batch
                    model_limit = embedding_fn.__self__.max_seq_length
                    
                SAFE_CHAR_LIMIT = int(model_limit * 4 * 0.95) # e.g. 512 -> ~1945 chars
            
            # Wrap embedding function to enforce truncation inside RAGAS
            from langchain_core.embeddings import Embeddings
            
            class SafeEmbeddingWrapper(Embeddings):
                def __init__(self, original_fn, limit):
                    self.fn = original_fn
                    self.limit = limit
                    self._has_warned = False
                    
                def _truncate(self, text: str) -> str:
                    if len(text) > self.limit:
                        if not self._has_warned:
                            logger.warning(f"RagasEvaluator: Truncating text > {self.limit} chars to fit embedding model ({model_limit} tokens). "
                                           "Consider using a larger chunker or limited context model.")
                            self._has_warned = True
                        return text[:self.limit]
                    return text
    
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    safe_texts = [self._truncate(t) for t in texts]
                    return self.fn(safe_texts)
                    
                def embed_query(self, text: str) -> List[float]:
                    return self.fn([self._truncate(text)])[0]
                    
            safe_embeddings = SafeEmbeddingWrapper(embedding_fn, SAFE_CHAR_LIMIT)
    
            metrics_to_run = [context_precision, context_recall]
            results = evaluate(
                dataset=dataset,
                metrics=metrics_to_run,
                llm=llm,
                embeddings=safe_embeddings  # Pass the safe wrapper
            )
            
            # Aggregate results
            final_metrics = {}
            for m in metrics_to_run:
                if m.name in results:
                    final_metrics[f"ragas.{m.name}"] = results[m.name]
                    
            logger.info(f"RagasEvaluator: Evaluation complete. Metrics: {final_metrics}")
            return final_metrics
        except Exception as e:
            logger.error(f"RagasEvaluator: Evaluation failed: {e}")
            return {"error": str(e)}
