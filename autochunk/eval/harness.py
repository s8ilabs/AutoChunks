
from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable
import random, time
from ..utils.text import split_sentences, whitespace_tokens
from ..utils.hashing import content_hash
from ..retrieval.in_memory import InMemoryIndex
from ..eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from ..eval.synthetic import SyntheticQAGenerator
from ..utils.logger import logger

class EvalHarness:
    def __init__(self, embedding_fn, k: int = 10):
        self.embedding = embedding_fn
        self.k = k
        self.generator = SyntheticQAGenerator()

    def build_synthetic_qa(self, docs: List[Dict], on_progress: Optional[Callable[[str], None]] = None) -> List[Dict]:
        qa = []
        rng = random.Random(42)
        for d in docs:
            sents = split_sentences(d["text"])[:20]
            # 1. Add standard paraphrased queries
            for s in sents[:2]:
                query = self.generator.generate_hard_query(s, on_progress)
                qa.append({
                    "id": content_hash(d["id"] + query),
                    "doc_id": d["id"],
                    "query": query,
                    "answer_span": s,
                })
            # 2. Add boundary-crossing queries (Advanced)
            if len(sents) > 2:
                boundary_qa = self.generator.generate_boundary_qa(d["id"], sents[:5], on_progress)
                qa.extend(boundary_qa)
        return qa

    def evaluate(self, chunks: List[Dict], qa: List[Dict]) -> Dict[str, Any]:
        # Build index
        # 1. Add distractor/noise chunks to ensure search isn't too trivial
        noise_chunks = []
        for i in range(20):
            noise_chunks.append({
                "id": f"noise_{i}", 
                "doc_id": "noise", 
                "text": f"This is some random distractor text about something unrelated {i} to increase complexity.",
                "meta": {}
            })
        
        all_eval_chunks = chunks + noise_chunks
        logger.info(f"EvalHarness: Encoding {len(all_eval_chunks)} chunks (including {len(noise_chunks)} noise)...")
        
        # Determine dynamic safety limit
        model_limit = 512 # Fallback
        
        # Check if it's Hashing (which has no limit)
        is_hashing = getattr(self.embedding, "name", "").startswith("hashing") or "HashingEmbedding" in str(type(self.embedding))
        
        if is_hashing:
             MAX_CHARS = 1_000_000 # Virtually infinite
             model_limit = 250_000
        else:
            if hasattr(self.embedding, "max_seq_length"):
                model_limit = self.embedding.max_seq_length
            elif hasattr(self.embedding, "__self__") and hasattr(self.embedding.__self__, "max_seq_length"):
                 # Handle bound methods
                model_limit = self.embedding.__self__.max_seq_length
            MAX_CHARS = int(model_limit * 4 * 0.95)
        
        has_warned = False
        def truncate(text: str) -> str:
            nonlocal has_warned
            if len(text) > MAX_CHARS:
                if not has_warned:
                    # Only warn if it's NOT hashing (since hashing truncation is rare/impossible with this high limit)
                    logger.warning(f"EvalHarness: Truncating chunks > {MAX_CHARS} chars to fit embedding model ({model_limit} tokens).")
                    has_warned = True
                return text[:MAX_CHARS]
            return text
        
        enc_start = time.time()
        try:
            vectors = self.embedding([truncate(c["text"]) for c in all_eval_chunks])
        except RuntimeError as e:
            if "expanded size" in str(e) or "512" in str(e):
                logger.error(f"EvalHarness: Embedding failed - some chunks exceed model's max token length. Truncating aggressively...")
                # Try with more aggressive truncation
                vectors = self.embedding([truncate(c["text"])[:1200] for c in all_eval_chunks])
            else:
                raise
        enc_time = time.time() - enc_start
        logger.info(f"EvalHarness: Encoding complete in {enc_time:.2f}s")
        
        index = InMemoryIndex(dim=len(vectors[0]))
        index.add(vectors, all_eval_chunks)

        mrr, ndcg, recall, covered = 0.0, 0.0, 0.0, 0
        
        # --- BATCH QUERY EVALUATION ---
        logger.info(f"EvalHarness: Encoding and searching {len(qa)} queries in batch mode...")
        # Reuse detection logic
        def truncate_q(text: str) -> str:
             return truncate(text) # Use the same robust logic and warning system

        query_texts = [truncate_q(item["query"]) for item in qa]
        try:
            query_vectors = self.embedding(query_texts)
        except RuntimeError as e:
            if "expanded size" in str(e):
                logger.error(f"EvalHarness: Query embedding failed - text too long. Truncating aggressively...")
                query_vectors = self.embedding([truncate_q(q)[:1000] for q in query_texts])
            else:
                raise
        
        # Batch search (using updated InMemoryIndex with batch support)
        batch_hits = index.search(query_vectors, top_k=self.k)
        
        for i, (item, hits) in enumerate(zip(qa, batch_hits)):
            target_doc = item["doc_id"].lower().replace("\\", "/")
            
            # --- TOKEN-LEVEL RECALL ---
            answer_tokens = set(whitespace_tokens(item["answer_span"].lower()))
            found_tokens = set()
            
            # --- RANKING & DCG RELEVANCE ---
            retrieved_rels = []
            has_perfect_match = False
            
            for rank, (idx, dist) in enumerate(hits):
                c = index.meta[idx]
                rel = 0.0
                
                # Check Document Match
                if c["doc_id"].lower().replace("\\", "/") == item["doc_id"].lower().replace("\\", "/"):
                    # Normalize whitespace for robust substring matching
                    chunk_text_norm = " ".join(c["text"].lower().split())
                    answer_norm = " ".join(item["answer_span"].lower().split())
                    
                    # 1. Full Answer Match (Highest Relevance)
                    if answer_norm in chunk_text_norm:
                        rel = 2.0
                        has_perfect_match = True
                        found_tokens.update(answer_tokens)
                    else:
                        # 2. Token Overlap Match (Partial Relevance)
                        chunk_tokens = set(chunk_text_norm.split())
                        overlap = answer_tokens.intersection(chunk_tokens)
                        if overlap:
                            rel = 1.0 + (len(overlap) / len(answer_tokens))
                            found_tokens.update(overlap)
                
                retrieved_rels.append(rel)

            # Score Aggregation
            # Relaxed Coverage: Allow non-exact but high-overlap matches (relevance > 1.5)
            # This accounts for markdown artifacts, minor cleaning diffs, etc.
            if has_perfect_match or any(r > 1.5 for r in retrieved_rels):
                covered += 1
            
            # MRR: Binary look (was there a perfect match in top-K?)
            binary_rels = [1 if r >= 2.0 else 0 for r in retrieved_rels]
            mrr += mrr_at_k(binary_rels, self.k)
            
            # nDCG: Uses the graduated relevance (0, 1.X, 2.0)
            ndcg += ndcg_at_k(retrieved_rels, self.k)
            
            # Recall: Percentage of total answer tokens covered by all top-K results
            current_recall = len(found_tokens) / len(answer_tokens) if answer_tokens else 0
            recall += current_recall

        n = max(1, len(qa))
        return {
            "mrr@k": mrr / n,
            "ndcg@k": ndcg / n,
            "recall@k": recall / n,
            "coverage": covered / n,
        }
