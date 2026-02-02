
import os
from autochunk import AutoChunker, AutoChunkConfig
from autochunk.utils.logger import logger

def run_pdf_demo():
    # 1. Setup local environment
    docs_dir = "./examples/pdf_docs"
    os.makedirs(docs_dir, exist_ok=True)
    
    logger.info("AutoChunks PDF  Demo")
    logger.info("-------------------------")
    
    # Note: Since this is a demo environment, we assume the user has 
    # put their PDF files in ./examples/pdf_docs
    
    if not any(f.endswith(".pdf") for f in os.listdir(docs_dir)):
        logger.warning(f"No PDFs found in {docs_dir}. Please drop some PDFs there to see  chunking.")
        return

    # 2. Configure AutoChunker
    # We use 'full' mode to explore the new LayoutAware and Docling strategies
    chunker = AutoChunker(
        mode="full",
        cache_dir=".ac_cache"
    )

    # 3. Optimize (This will now include layout-aware candidates automatically)
    logger.info("Optimizing for PDF content...")
    best_plan, report = chunker.optimize(
        documents=docs_dir,
    )

    logger.success(f"Best strategy for these PDFs: {best_plan.generator_pipeline['name']}")
    
    # 4. Apply
    chunks = best_plan.apply(docs_dir, chunker)
    logger.info(f"Generated {len(chunks)} chunks with layout preservation.")
    
    if chunks:
        sample = chunks[0]
        logger.info(f"Sample Chunk Meta: {sample['meta']}")
        logger.debug(f"Sample Text Start: {sample['text'][:200]}...")

if __name__ == "__main__":
    run_pdf_demo()
