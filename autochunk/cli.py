
import argparse
import sys
import os
import json
import yaml
from typing import List, Optional

from .autochunker import AutoChunker
from .config import AutoChunkConfig, EvalConfig, ProxyConfig, NetworkConfig, RagasConfig
from .storage.plan import Plan
from .utils.logger import logger

def cmd_optimize(args):
    """Run the optimization search from CLI."""
    logger.info(f"Starting CLI Optimization on: {args.docs}")
    
    cfg = AutoChunkConfig(
        mode=args.mode,
        embedding_provider=args.embedding_provider,
        embedding_model_or_path=args.embedding_model,
        cache_dir=args.cache_dir
    )
    
    # Optional override for proxy
    if args.no_proxy:
        cfg.proxy_config.enabled = False
    
    chunker = AutoChunker(
        mode=cfg.mode,
        eval_config=EvalConfig(objective=args.objective),
        embedding_provider=cfg.embedding_provider,
        embedding_model_or_path=cfg.embedding_model_or_path,
        cache_dir=cfg.cache_dir
    )
    
    # Enable RAGAS if flag is set
    if args.analyze_ragas:
        chunker.cfg.ragas_config.enabled = True
        if hasattr(args, 'ragas_llm_provider') and args.ragas_llm_provider:
            chunker.cfg.ragas_config.llm_provider = args.ragas_llm_provider
        if hasattr(args, 'ragas_llm_model') and args.ragas_llm_model:
            chunker.cfg.ragas_config.llm_model = args.ragas_llm_model
        logger.info(f"RAGAS enabled. LLM Provider: {chunker.cfg.ragas_config.llm_provider}")
    
    def on_progress(msg, step=None):
        prefix = f"[Stage {step}] " if step else ""
        logger.info(f"{prefix}{msg}")

    plan, report = chunker.optimize(
        documents=args.docs,
        on_progress=on_progress
    )
    
    # Save the plan
    Plan.write(args.out, plan)
    logger.success(f"Optimization complete! Winning plan saved to: {args.out}")
    
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Full metrics report saved to: {args.report}")

def cmd_apply(args):
    """Apply a winning plan to a corpus."""
    logger.info(f"Applying plan {args.plan} to docs: {args.docs}")
    
    plan = Plan.read(args.plan)
    
    # Initialize chunker with plan's embedding settings
    chunker = AutoChunker(
        embedding_provider=plan.embedding.get("name"),
        embedding_model_or_path=plan.embedding.get("model")
    )
    
    chunks = plan.apply(args.docs, chunker)
    
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    logger.success(f"Successfully created {len(chunks)} chunks. Selection saved to: {args.out}")

def cmd_serve(args):
    """Launch the Dashboard."""
    import uvicorn
    from .web.server import app
    logger.info(f"Launching AutoChunks Dashboard on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

def main():
    parser = argparse.ArgumentParser(
        description="AutoChunks: Autonomous Retrieval Optimization for RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Optimize command
    opt_p = subparsers.add_parser("optimize", help="Search for the best chunking strategy")
    opt_p.add_argument("--docs", required=True, help="Path to document folder")
    opt_p.add_argument("--mode", default="light", choices=["light", "full"], help="Evaluation Depth: Controls synthetic QA sampling density")
    opt_p.add_argument("--objective", default="balanced", choices=["balanced", "quality", "cost", "latency", "metric_only"], help="Optimization objective")
    opt_p.add_argument("--embedding-provider", default="hashing", help="Embedding provider")
    opt_p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Model name or path")
    opt_p.add_argument("--out", default="best_plan.yaml", help="Output path for the winning plan")
    opt_p.add_argument("--report", help="Output path for the full JSON metrics report")
    opt_p.add_argument("--cache-dir", default=".ac_cache", help="Cache directory")
    opt_p.add_argument("--no-proxy", action="store_true", help="Disable representative sampling (process all docs)")
    opt_p.add_argument("--analyze-ragas", action="store_true", help="Enable RAGAS LLM-based evaluation metrics (Context Precision/Recall)")
    opt_p.add_argument("--ragas-llm-provider", default="auto", choices=["auto", "openai", "ollama", "huggingface"], help="LLM provider for RAGAS (auto detects OpenAI key or Ollama)")
    opt_p.add_argument("--ragas-llm-model", default=None, help="Model name for RAGAS LLM (e.g., gpt-4o-mini, llama3.2)")

    # Apply command
    app_p = subparsers.add_parser("apply", help="Execute a saved plan on a corpus")
    app_p.add_argument("--plan", required=True, help="Path to the .yaml plan file")
    app_p.add_argument("--docs", required=True, help="Path to documents to chunk")
    app_p.add_argument("--out", default="chunks.json", help="Output path for processed chunks")

    # Serve command
    srv_p = subparsers.add_parser("serve", help="Start the Web Dashboard")
    srv_p.add_argument("--host", default="0.0.0.0", help="Host address")
    srv_p.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    if args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "apply":
        cmd_apply(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
