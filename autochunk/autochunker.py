
from __future__ import annotations
import time, json, os, random, math
import concurrent.futures
from opentelemetry import context
from typing import List, Dict, Any, Tuple, Optional, Callable
from .config import AutoChunkConfig
from .storage.cache import Cache
from .storage.plan import Plan
from .utils.hashing import content_hash
from .utils.io import load_documents
from .utils.logger import logger, current_job_id
from .utils.telemetry import init_telemetry, get_tracer
from .chunkers.fixed_length import FixedLengthChunker
from .chunkers.sentence_aware import SentenceAwareChunker
from .chunkers.recursive_character import RecursiveCharacterChunker
from .chunkers.semantic_local import SemanticLocalChunker
from .chunkers.parent_child import ParentChildChunker
from .chunkers.layout_aware import LayoutAwareChunker
from .chunkers.hybrid_semantic_stat import HybridSemanticStatChunker
from .chunkers.html_section import HTMLSectionChunker
from .chunkers.bridges.langchain.recursive import LangChainRecursiveBridge
from .chunkers.bridges.langchain.character import LangChainCharacterBridge
from .chunkers.bridges.langchain.markdown import LangChainMarkdownBridge
from .chunkers.bridges.langchain.token import LangChainTokenBridge
from .chunkers.bridges.langchain.python import LangChainPythonBridge
from .chunkers.bridges.langchain.html import LangChainHTMLBridge
from .chunkers.bridges.langchain.json import LangChainJSONBridge
from .embedding.hashing import HashingEmbedding
from .eval.harness import EvalHarness
from .eval.ragas_eval import RagasEvaluator
# Quality Layer - Post-Processing Pipeline
from .quality.post_processor import apply_post_processing, NATIVE_CHUNKERS

tracer = get_tracer(__name__)

GENERATOR_REGISTRY = {
    "fixed_length": FixedLengthChunker(),
    "sentence_aware": SentenceAwareChunker(),
    "recursive_character": RecursiveCharacterChunker(),
    "semantic_local": SemanticLocalChunker(),
    "parent_child": ParentChildChunker(),
    "layout_aware": LayoutAwareChunker(),
    "hybrid_semantic_stat": HybridSemanticStatChunker(),
    "html_section": HTMLSectionChunker(),
    # Framework Bridges
    "langchain_recursive": LangChainRecursiveBridge(),
    "langchain_character": LangChainCharacterBridge(),
    "langchain_markdown": LangChainMarkdownBridge(),
    "langchain_token": LangChainTokenBridge(),
    "langchain_python": LangChainPythonBridge(),
    "langchain_html": LangChainHTMLBridge(),
    "langchain_json": LangChainJSONBridge(),
}

class AutoChunker:
    def __init__(self, eval_config=None, retrieval_strategy=None, proxy_config=None, ragas_config=None, mode="light", 
                 cache_dir=".ac_cache", metadata_enrichment=None, 
                 embedding_provider=None, embedding_model_or_path=None, embedding_api_key=None, network_config=None,
                 telemetry_enabled: bool = False,
                 # Post-Processing Options (AutoChunks native chunkers only)
                 enable_dedup: bool = False,
                 enable_overlap_opt: bool = False,
                 dedup_threshold: float = 0.98,
                 overlap_tokens: int = 50):
        self.cfg = AutoChunkConfig()
        if eval_config: self.cfg.eval_config = eval_config
        if retrieval_strategy: self.cfg.retrieval_strategy = retrieval_strategy
        if proxy_config: self.cfg.proxy_config = proxy_config
        if ragas_config: self.cfg.ragas_config = ragas_config
        if embedding_provider: self.cfg.embedding_provider = embedding_provider
        if embedding_model_or_path: self.cfg.embedding_model_or_path = embedding_model_or_path
        if embedding_api_key: self.cfg.embedding_api_key = embedding_api_key
        if network_config: self.cfg.network = network_config
        self.cfg.telemetry_enabled = telemetry_enabled
        self.cfg.mode = mode
        self.cfg.cache_dir = cache_dir
        if metadata_enrichment: self.cfg.metadata_enrichment = metadata_enrichment
        
        # Post-processing config
        self.enable_dedup = enable_dedup
        self.enable_overlap_opt = enable_overlap_opt
        self.dedup_threshold = dedup_threshold
        self.overlap_tokens = overlap_tokens
        
        init_telemetry(enabled=self.cfg.telemetry_enabled)

    def optimize(self, documents: List[Dict]|str, embedding_fn=None, retriever="in_memory", 
                 framework="langchain", golden_qa=None, candidate_names: Optional[List[str]] = None,
                 sweep_params: Optional[Dict[str, List]] = None,
                 on_progress: Optional[Callable[[str, int], None]] = None,
                 on_result: Optional[Callable[[Dict], None]] = None):
        
        if on_progress: on_progress(f"Scanning documents in {documents if isinstance(documents, str) else 'memory'}...", step=1)
        
        # Load docs if path provided
        if isinstance(documents, str):
            # Check if any selected candidate needs high-fidelity (Markdown) extraction
            high_fidelity = any(c in (candidate_names or []) for c in ["layout_aware", "hybrid_semantic_stat"])
            if not candidate_names: high_fidelity = True # Default to best quality if none specified
            
            docs = load_documents(
                documents, 
                on_progress=lambda m: on_progress(m, 1) if on_progress else None,
                high_fidelity=high_fidelity
            )
        else:
            docs = documents
        
        # 1. Compute Document Hashes
        if on_progress: on_progress("Computing document fingerprints (hashing)...", step=1)
        for d in docs:
            if "hash" not in d:
                d["hash"] = content_hash(d["text"])
        logger.info(f"Hashed {len(docs)} documents.")
        
        corpus_hash = content_hash("".join(sorted(d["hash"] for d in docs)))
        logger.info(f"Corpus Fingerprint: {corpus_hash[:12]}...")

        with tracer.start_as_current_span("optimize") as span:
            span.set_attribute("corpus_hash", corpus_hash)
            span.set_attribute("mode", self.cfg.mode)
            
            # Embedding setup (Pluggable Architecture)
            if embedding_fn is None:
                from .embedding import get_encoder
                if on_progress: on_progress(f"Initializing {self.cfg.embedding_provider} encoder...", step=1)
                encoder = get_encoder(
                    provider=self.cfg.embedding_provider, 
                    model_name=self.cfg.embedding_model_or_path,
                    api_key=self.cfg.embedding_api_key,
                    cache_folder=self.cfg.network.local_models_path,
                    trusted_orgs=self.cfg.network.trusted_orgs
                )
                logger.info(f"Initialized {self.cfg.embedding_provider} encoder: {encoder.model_name}")
                if on_progress: on_progress(f"Encoder Ready: {self.cfg.embedding_provider}", step=1)
                embedding_fn = encoder.embed_batch
                span.set_attribute("embedding.provider", self.cfg.embedding_provider)
                span.set_attribute("embedding.model", self.cfg.embedding_model_or_path)
            
            harness = EvalHarness(embedding_fn, k=self.cfg.eval_config.k)
            
            # 2. Representative Proxy Sampling (Optimized for Scale)
            docs_sorted = sorted(docs, key=lambda x: x["id"])
            proxy_docs = docs_sorted
            if self.cfg.proxy_config.enabled or len(docs_sorted) > 10:
                n_samples = max(2, int(len(docs_sorted) * (self.cfg.proxy_config.proxy_percent / 100.0)))
                if self.cfg.mode == "light":
                    n_samples = min(n_samples, 5)
                
                random.seed(42)  # Deterministic sampling
                proxy_docs = random.sample(docs_sorted, min(len(docs_sorted), n_samples))
                logger.info(f"Proxy Strategy: Optimizing on {len(proxy_docs)} representative docs (Total: {len(docs_sorted)})")

            with tracer.start_as_current_span("build_synthetic_qa"):
                cache = Cache(self.cfg.cache_dir)
                qa = []
                
                if golden_qa:
                    qa = golden_qa
                    logger.info(f"Using provided golden QA ({len(qa)} pairs)")
                else:
                    docs_needing_qa = []
                    for d in proxy_docs:
                        doc_qa_key = f"qa_doc_{d['hash']}"
                        cached_doc_qa = cache.get_json(doc_qa_key)
                        if cached_doc_qa:
                            # Re-bind doc_id to the current session's path to ensure cache portability
                            for q in cached_doc_qa:
                                q["doc_id"] = d["id"]
                            qa.extend(cached_doc_qa)
                        else:
                            docs_needing_qa.append(d)
                    
                    if docs_needing_qa:
                        if on_progress: on_progress(f"Generating synthetic QA for {len(docs_needing_qa)} files...", step=2)
                        for i, d in enumerate(docs_needing_qa):
                            if on_progress: on_progress(f"Analyzing document {i+1}/{len(docs_needing_qa)} [{os.path.basename(d['id'])}]...", step=2)
                            doc_qa = harness.build_synthetic_qa([d], lambda m: on_progress(m, 2) if on_progress else None)
                            doc_qa_key = f"qa_doc_{d['hash']}"
                            cache.set_json(doc_qa_key, doc_qa)
                            qa.extend(doc_qa)
                            
                        logger.info(f"Generated QA for {len(docs_needing_qa)} documents. Total QA pool: {len(qa)}")
                        if on_progress: on_progress(f"QA Generation complete ({len(qa)} pairs total)", step=2)
                    else:
                        logger.info(f"Cache Hit: Reusing {len(qa)} QA pairs for sampled docs")
                        if on_progress: on_progress(f"Reusing {len(qa)} cached QA pairs", step=2)
                
                span.set_attribute("num_qa_pairs", len(qa))

            # Candidate grid
            all_candidates = [
                ("fixed_length", {"base_token_size": 512, "overlap": 64}),
                ("recursive_character", {"base_token_size": 512}),
                ("sentence_aware", {"base_token_size": 512}),
                ("semantic_local", {"threshold_percentile": 0.8}),
                ("hybrid_semantic_stat", {"alpha": 0.7, "beta": 0.3}),
                ("parent_child", {"parent_size": 1000, "child_size": 200}),
            ]
            
            # Layout-aware candidates for complex formats
            has_rich_docs = any(d.get("ext") in [".pdf", ".md", ".html", ".htm"] for d in proxy_docs)
            if has_rich_docs:
                all_candidates.append(("layout_aware", {"base_token_size": 512}))
            
            # HTML Section Chunker (Native)
            has_html = any(d.get("ext") in [".html", ".htm"] for d in proxy_docs)
            if has_html:
                all_candidates.append(("html_section", {"base_token_size": 512}))

            # Add Framework Bridges to candidates if they exist
            all_candidates.extend([
                ("langchain_recursive", {"base_token_size": 512, "overlap": 64}),
                ("langchain_character", {"base_token_size": 512, "overlap": 64}),
                ("langchain_markdown", {"base_token_size": 512, "overlap": 64}),
                ("langchain_token", {"base_token_size": 512, "overlap": 64}),
                ("langchain_python", {"base_token_size": 512, "overlap": 64}),
                ("langchain_html", {"base_token_size": 512, "overlap": 64}),
                ("langchain_json", {"base_token_size": 512, "overlap": 64}),
            ])
                
            if candidate_names:
                base_candidates = [c for c in all_candidates if c[0] in candidate_names]
            else:
                base_candidates = all_candidates

            if not base_candidates:
                base_candidates = all_candidates[:3]

            # --- Hyperparameter Sweep Expansion ---
            candidates = []
            if sweep_params and (sweep_params.get("chunk_sizes") or sweep_params.get("overlap_ratios")):
                logger.info(f"Applying Hyperparameter Sweep: {sweep_params}")
                
                sizes = sweep_params.get("chunk_sizes", [512])
                ratios = sweep_params.get("overlap_ratios", [0.125])
                
                for name, default_params in base_candidates:
                    # Check if this candidate supports sizing
                    is_sizable = "base_token_size" in default_params or name.startswith("langchain_") or name in ["fixed_length", "recursive_character", "sentence_aware", "layout_aware", "html_section"]
                    
                    if is_sizable and name != "semantic_local": 
                        for s in sizes:
                            for r in ratios:
                                p = default_params.copy()
                                p["base_token_size"] = s
                                # Calculate overlap
                                overlap = int(s * r)
                                
                                # Apply to params if appropriate for the chunker
                                supported_overlap = [
                                    "fixed_length", "recursive_character", "sentence_aware", 
                                    "layout_aware", "html_section", "langchain_recursive", 
                                    "langchain_token", "langchain_character", "langchain_markdown", 
                                    "langchain_python", "langchain_html", "langchain_json"
                                ]
                                if "overlap" in p or name in supported_overlap:
                                    p["overlap"] = overlap
                                elif name == "parent_child":
                                    # parent_child usually has fixed parent/child sizes, skipping sweep for now
                                    continue 
                                
                                # Use a more descriptive name for the variants
                                variant_name = f"{name} ({s}|{int(r*100)}%)" if len(sizes) > 1 or len(ratios) > 1 else name
                                # We pass (base_name, display_name, params)
                                candidates.append((name, variant_name, p))
                    else:
                        candidates.append((name, name, default_params))
            else:
                # base_candidates are already (name, params) usually, but we need (base_name, display_name, params)
                candidates = [(c[0], c[0], c[1]) for c in base_candidates]

            if on_progress: on_progress(f"Starting parallel evaluation of {len(candidates)} candidates...", step=3)
            best = None
            best_metrics = None
            best_score = -1.0
            reports = []

            logger.info(f"Starting Parallel Optimization across {len(candidates)} candidates...")
            
            parent_ctx = context.get_current()
            job_id_ctx = current_job_id.get()
            # Safety: Cap workers to 4 to prevent GIL deadlock, further reduce for local models to prevent resource contention
            safe_workers = min(4, self.cfg.parallel.embedding_concurrency or 4)
            if self.cfg.embedding_provider == "local":
                 safe_workers = min(safe_workers, 2)
                 logger.info(f"Using reduced concurrency ({safe_workers} workers) for local embeddings to optimize resource usage.")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=safe_workers) as executor:
                futures = {
                    executor.submit(self._eval_candidate, cand, proxy_docs, qa, embedding_fn, retriever, parent_ctx, job_id_ctx): cand 
                    for cand in candidates
                }
                
                for future in concurrent.futures.as_completed(futures):
                    # cand_info is (base_name, display_name, params)
                    cand_info = futures[future]
                    try:
                        logger.info(f"Checking result for candidate: {cand_info[1]}...")
                        # Safety: Add 5-minute timeout per candidate
                        name, params, metrics, chunks = future.result(timeout=300)
                        logger.info(f"Retrieved result for {name}. Processing...")
                        # Calculate the multi-objective score
                        current_score = self._calculate_score(metrics, self.cfg.eval_config.objective)
                        metrics["objective_score"] = current_score

                        # Create result object
                        result_entry = {
                            "name": name,
                            "params": params,
                            "metrics": metrics,
                            "score": current_score,
                            "is_partial": True
                        }
                        
                        # Streaming Callback
                        if on_result:
                            on_result(result_entry)

                        # Include a small sample of chunks for Visual Fidelity Inspector 
                        from .utils.text import count_tokens
                        chunk_samples = []
                        for c in chunks[:3]:
                            chunk_samples.append({
                                "text": c["text"],
                                "meta": c["meta"],
                                "tokens": count_tokens(c["text"])
                            })
                        
                        report_entry = {
                            "name": name,
                            "params": params,
                            "metrics": metrics,
                            "chunk_samples": chunk_samples
                        }
                        
                        logger.info(f"Evaluated {name}: Score {current_score:.4f}")
                        
                        # Fix for 0.0000 logs: Ensure we use the same key mapping as _calculate_score
                        # Config defines e.g. 'mrr@10', but harness returns 'mrr@k'
                        primary_display_key = self.cfg.eval_config.metrics[0]
                        key_map = {"ndcg@10": "ndcg@k", "mrr@10": "mrr@k", "recall@50": "recall@k"}
                        target_key = key_map.get(primary_display_key, primary_display_key)
                        
                        metric_val = metrics.get(target_key, 0)
                        
                        # Diagnostic logging if metric is unexpectedly zero
                        if metric_val == 0 and current_score > 0:
                            logger.debug(f"Metric lookup for {primary_display_key} -> {target_key} returned 0. Available keys: {list(metrics.keys())}")
                        
                        if on_progress: on_progress(f"Evaluated {name}: {metric_val:.4f}", step=3)

                        if (best is None) or (current_score > best_score):
                            best = (name, params) # Chunks are not stored in 'best' anymore
                            best_metrics = metrics
                            best_score = current_score
                            logger.success(f"New Leader: {name} (Score: {best_score:.4f}, {list(metrics.keys())[0]}: {list(metrics.values())[0]:.4f})")
                        
                        reports.append(report_entry)
                        
                    except Exception as e:
                        logger.error(f"Candidate evaluation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        if on_progress: on_progress(f"Candidate failed", step=3)
            
            if not best:
                raise RuntimeError("Optimization failed: No valid candidates were successfully evaluated.")

            name, params = best # Chunks are not part of 'best' anymore
            # Re-run the best candidate to get its chunks for the final plan, if needed
            # Or, if chunks are only for reporting, we can use the samples from reports.
            # For now, assuming chunks are not needed for the final plan object directly.
            plan = Plan(
                id=content_hash(corpus_hash + name + json.dumps(params)),
                corpus_hash=corpus_hash,
                generator_pipeline={"name": name, "params": params},
                metrics=best_metrics,
                embedding={"name": self.cfg.embedding_provider, "model": self.cfg.embedding_model_or_path}
            )
            report = {"candidates": reports, "selected": {"name": name, "params": params, "metrics": best_metrics}}
            return plan, report

    def apply_with_generator(self, documents: str|List[Dict], gen_name: str, params: Dict) -> List[Dict]:
        if isinstance(documents, str):
            docs = load_documents(documents)
        else:
            docs = documents
        gen = GENERATOR_REGISTRY[gen_name]
        all_chunks = []
        p = params.copy()
        if gen_name in ["semantic_local", "hybrid_semantic_stat"] and "embedding_fn" not in p:
            from .embedding import get_encoder
            encoder = get_encoder(provider=self.cfg.embedding_provider, model_name=self.cfg.embedding_model_or_path, cache_folder=self.cfg.network.local_models_path, trusted_orgs=self.cfg.network.trusted_orgs)
            p["embedding_fn"] = encoder.embed_batch
        
        # Determine if this is a bridge chunker
        is_bridge = gen_name.startswith("langchain_")
        
        for d in docs:
            p["local_models_path"] = self.cfg.network.local_models_path
            doc_meta = d.get("metadata", {})
            
            # Use raw_text for bridges, processed text for native chunkers
            if is_bridge:
                doc_text = d.get("raw_text", d["text"])
            else:
                doc_text = d["text"]
            
            try:
                for ch in gen.chunk(d["id"], doc_text, **p):
                    # Combine doc metadata with chunk-specific metadata
                    combined_meta = {**doc_meta, **ch.meta}
                    all_chunks.append({
                        "id": ch.id, 
                        "doc_id": d["id"], 
                        "text": ch.text, 
                        "meta": combined_meta
                    })
            except Exception as e:
                logger.warning(f"Chunker {gen_name} failed on doc {d['id']}: {e}")
        return all_chunks

    def _calculate_score(self, metrics: Dict[str, Any], objective: str) -> float:
        """
         Weighted Scorer based on the target objective.
        Weights:
        - Quality: nDCG@k (Retrieval Precision)
        - Coverage: Percentage of queries with a perfect match
        - Efficiency: Penalty for excessive chunk counts
        """
        # Determine the anchor quality metric based on user selection
        primary_key = self.cfg.eval_config.metrics[0]
        key_map = {"ndcg@10": "ndcg@k", "mrr@10": "mrr@k", "recall@50": "recall@k"}
        target_key = key_map.get(primary_key, "ndcg@k")
        
        q = metrics.get(target_key, 0)
        c = metrics.get("coverage", 0)
        m = metrics.get("mrr@k", 0)
        count = metrics.get("count", 1)
        
        # Logarithmic penalty for chunk count
        cost_penalty = 0.05 * math.log10(max(1, count))

        if objective == "quality":
            return q * 0.9 + m * 0.1
        elif objective == "cost":
            # Cost Optimized: Heavy penalty on count, but quality still matters
            # Using inverse log scale so massive counts drop score towards 0
            efficiency_score = 1.0 / (1.0 + 0.2 * math.log10(max(1, count)))
            return q * 0.4 + efficiency_score * 0.6
        elif objective == "latency":
            # Latency Focus: MRR is king (finds answer fast at rank 1)
            return m * 0.8 + q * 0.2
        else: # "balanced"
            # Balanced: Quality (nDCG) + Coverage (Reliability) - mild cost penalty
            # Adjusted to be less punishing for large-but-good indices
            base_score = (q * 0.6) + (m * 0.2) + (c * 0.2)
            
            # Capped cost penalty: Max deduction is 15% for huge indices
            # log10(1000) = 3 -> 0.09 penalty
            # log10(10000) = 4 -> 0.12 penalty
            normalized_cost_penalty = 0.03 * math.log10(max(1, count))
            
            return max(0.0, base_score - normalized_cost_penalty)

    def _eval_candidate(self, cand: Tuple[str, str, Dict], docs: List[Dict], qa: List[Dict], embedding_fn: Any, retriever: str, otel_context=None, job_id_ctx=None) -> Tuple[str, Dict, Dict, List]:
        if otel_context:
            context.attach(otel_context)
        if job_id_ctx:
            current_job_id.set(job_id_ctx)
        base_name, display_name, params = cand
        with tracer.start_as_current_span(f"candidate.{display_name}") as cspan:
            cspan.set_attribute("params", json.dumps(params))
            gen = GENERATOR_REGISTRY[base_name]
            p = params.copy()
            p["local_models_path"] = self.cfg.network.local_models_path
            if base_name in ["semantic_local", "hybrid_semantic_stat"]:
                p["embedding_fn"] = embedding_fn
            harness = EvalHarness(embedding_fn, k=self.cfg.eval_config.k)
            
            # ═══════════════════════════════════════════════════════════════════
            # FAIR EVALUATION: Use different text versions for different chunkers
            # - Native AutoChunks: Gets processed/optimized text (our value-add)  
            # - LangChain Bridges: Gets raw text (fair comparison, no preprocessing)
            # ═══════════════════════════════════════════════════════════════════
            is_native = base_name in NATIVE_CHUNKERS
            is_bridge = base_name.startswith("langchain_")
            
            if is_bridge:
                logger.info(f"[{display_name}] Starting chunking (RAW text mode)...")
                # DIAGNOSTIC: Log exact parameters passed to bridge
                logger.debug(f"[{display_name}] Parameters: size={params.get('base_token_size')}, overlap={params.get('overlap')}")
            else:
                logger.info(f"[{display_name}] Starting chunking (Processed text mode)...")
            
            start_time = time.time()
            chunks = []
            for i, d in enumerate(docs):
                try:
                    # Use raw_text for bridges, processed text for native chunkers
                    if is_bridge:
                        # Bridges get raw text - no preprocessing advantage
                        doc_text = d.get("raw_text", d["text"])
                    else:
                        # Native chunkers get processed text (our value-add)
                        doc_text = d["text"]
                    
                    logger.debug(f"[{display_name}] Chunking document {i+1}/{len(docs)}...")
                    for ch in gen.chunk(d["id"], doc_text, **p):
                        chunks.append({"id": ch.id, "doc_id": d["id"], "text": ch.text, "meta": ch.meta})
                except Exception as e:
                    logger.warning(f"[{display_name}] Chunker failed on doc {d['id']}: {e}")
            
            chunking_time = time.time() - start_time
            logger.info(f"[{display_name}] Chunking complete: {len(chunks)} chunks in {chunking_time:.2f}s")
            
            if not chunks:
                logger.error(f"[{display_name}] returned zero chunks.")
                raise ValueError(f"Chunker {display_name} returned zero chunks.")
            
            # ═══════════════════════════════════════════════════════════════════
            # POST-PROCESSING & QUALITY SCORING
            # ═══════════════════════════════════════════════════════════════════
            quality_metrics = {}
            
            try:
                logger.info(f"[{display_name}] Starting post-processing...")
                pp_start = time.time()
                # Always call post-processing. The processor internally checks 'chunker_name'
                # and skips modifications for non-native chunkers, but still returns quality scores.
                processed_chunks, quality_metrics = apply_post_processing(
                    chunks=chunks,
                    chunker_name=base_name, # Use base_name for internal logic
                    embedding_fn=embedding_fn,
                    enable_dedup=self.enable_dedup,
                    enable_overlap=self.enable_overlap_opt,
                    dedup_threshold=self.dedup_threshold,
                    overlap_tokens=self.overlap_tokens
                )
                
                pp_time = time.time() - pp_start
                if quality_metrics.get("dedup_removed", 0) > 0:
                    logger.info(f"[{display_name}] Post-processing: Removed {quality_metrics['dedup_removed']} duplicate chunks in {pp_time:.2f}s")
                else:
                    logger.info(f"[{display_name}] Post-processing complete in {pp_time:.2f}s")
                
                chunks = processed_chunks
            except Exception as e:
                logger.warning(f"[{display_name}] Post-processing/Scoring failed: {e}")
            
            # Evaluate with standard metrics (nDCG, MRR, etc.)
            logger.info(f"[{display_name}] Starting evaluation against {len(qa)} queries...")
            eval_start = time.time()
            metrics = harness.evaluate(chunks, qa)
            eval_time = time.time() - eval_start
            logger.info(f"[{display_name}] Evaluation complete in {eval_time:.2f}s")
            
            # Add explicit Chunk Count (Critical for UI)
            metrics["count"] = len(chunks)
            
            # Add quality metrics (fair: calculated for both native and bridge)
            if quality_metrics:
                metrics["avg_quality_score"] = quality_metrics.get("avg_quality_score", 0)
                metrics["post_processed"] = quality_metrics.get("post_processing_applied", False)
                metrics["dedup_removed"] = quality_metrics.get("dedup_removed", 0)
                if "quality_dimensions" in quality_metrics:
                    metrics["quality_coherence"] = quality_metrics["quality_dimensions"].get("coherence", 0)
                    metrics["quality_completeness"] = quality_metrics["quality_dimensions"].get("completeness", 0)
                    metrics["quality_density"] = quality_metrics["quality_dimensions"].get("density", 0)
            
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    cspan.set_attribute(f"metrics.{k}", v)

            # RAGAS Evaluation (Optional, Plug-and-Play)
            if self.cfg.ragas_config.enabled:
                logger.info(f"[{display_name}] Starting RAGAS evaluation...")
                try:
                    ragas_eval = RagasEvaluator(self.cfg.ragas_config)
                    # Note: RagasEvaluator currently needs 'retrieved_ids' in QA items.
                    # Since we are decoupling, we will make RagasEvaluator handle retrieval internally 
                    # OR we need to perform retrieval here to pass it.
                    # For perfectly clean separation, RagasEvaluator shoud accept embedding_fn to run search.
                    # Let's pass the embedding_fn to RagasEvaluator's run method if we update it.
                    # But wait, RagasEvaluator logic we wrote assumes 'retrieved_ids'.
                    # We need to bridge this gap.
                    # Let's simple utilize the harness to get hits first?
                    # The cleanest way without changing `evaluate` signature is to call a header helper
                    # or just rely on RagasEvaluator doing the search (which we need to implement).
                    # Let's assume for this step we update RagasEvaluator to accept embedding_fn.
                    
                    ragas_metrics = ragas_eval.run(chunks, qa, embedding_fn=embedding_fn)
                    if ragas_metrics:
                       metrics.update(ragas_metrics)
                       logger.success(f"[{display_name}] RAGAS Metrics: {ragas_metrics}")
                       for k, v in ragas_metrics.items():
                            cspan.set_attribute(f"metrics.ragas.{k}", v)

                except Exception as e:
                    logger.warning(f"[{display_name}] RAGAS Evaluation failed: {e}")

            total_time = time.time() - start_time
            logger.success(f"[{display_name}] Candidate evaluation finished in {total_time:.2f}s")
            return display_name, params, metrics, chunks
