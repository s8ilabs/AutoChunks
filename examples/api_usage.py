
import os
from autochunk import AutoChunker
from autochunk.config import EvalConfig, RetrievalStrategy
from autochunk.embedding.hashing import HashingEmbedding
from autochunk.utils.logger import logger

def run_api_demo():
    """
    To see the traces in Arize Phoenix:
    1. Install it: pip install arize-phoenix
    2. Start the server: python -m phoenix.server.main serve
    3. Traces will appear at: http://localhost:6006
    """
    
    # 1. Prepare your documents
    docs_path = os.path.abspath("examples/sample_docs")
    
    # 2. Configure the optimizer
    eval_cfg = EvalConfig(
        metrics=["mrr@10", "ndcg@10"],
        k=5,
        objective="balanced"
    )
    
    retrieval_strat = RetrievalStrategy(
        type="standard",
        child_token_size=128
    )
    
    # 3. Initialize the AutoChunker
    # Telemetry is initialized automatically in the constructor
    chunker = AutoChunker(
        eval_config=eval_cfg,
        retrieval_strategy=retrieval_strat,
        mode="light"
    )
    
    # 4. Define a custom embedding function (optional)
    emb = HashingEmbedding(dim=128)
    
    logger.info("Starting demo run...")
    
    # 5. Run the optimization
    # Traces for each candidate will be sent to Phoenix via OpenTelemetry
    best_plan, report = chunker.optimize(
        documents=docs_path,
        embedding_fn=emb.embed_batch,
        retriever="in_memory"
    )
    
    # 6. Apply the plan to get chunks
    chunks = best_plan.apply(docs_path, chunker)
    
    logger.success(f"Final Selection: {best_plan.generator_pipeline['name']}")
    logger.info(f"Total chunks created: {len(chunks)}")

if __name__ == "__main__":
    run_api_demo()

