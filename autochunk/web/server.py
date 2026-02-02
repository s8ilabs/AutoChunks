
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr
from typing import List, Dict, Any, Optional
import os
import json
import asyncio
import time
from autochunk import AutoChunker, AutoChunkConfig, EvalConfig, RetrievalStrategy, ProxyConfig, NetworkConfig, RagasConfig
from autochunk.utils.logger import logger, jobs_data, current_job_id

app = FastAPI(title="AutoChunks Dashboard")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job data is now imported from .utils.logger for synchronization

class OptimizeRequest(BaseModel):
    documents_path: str
    mode: str = "light"
    objective: str = "balanced"
    metrics: List[str] = ["ndcg@10", "mrr@10"]
    proxy_enabled: bool = True
    proxy_percent: int = 10
    embedding_provider: str = "hashing"
    embedding_model_or_path: str = "BAAI/bge-small-en-v1.5"
    selected_candidates: Optional[List[str]] = None
    local_models_path: Optional[str] = None
    embedding_api_key: Optional[SecretStr] = None
    telemetry_enabled: bool = False
    sweep_params: Optional[Dict[str, List]] = None
    analyze_ragas: bool = False
    ragas_llm_provider: str = "auto"  # auto|openai|ollama|huggingface
    ragas_llm_model: Optional[str] = None
    ragas_api_key: Optional[SecretStr] = None

def add_log(job_id: str, msg: str, is_success: bool = False):
    timestamp = time.strftime("%H:%M:%S")
    entry = {"time": timestamp, "msg": msg, "success": is_success}
    jobs_data[job_id]["logs"].append(entry)

def run_optimization_sync(job_id: str, req: OptimizeRequest):
    # Standardize logs via loguru context
    with logger.contextualize(job_id=job_id):
        current_job_id.set(job_id)
        try:
            # Step 1: Setup
            jobs_data[job_id]["step"] = 1
            add_log(job_id, f"Configuring {req.embedding_provider.upper()} engine ({req.mode} mode)...")
            
            emb_key = req.embedding_api_key.get_secret_value() if req.embedding_api_key else None
            ragas_key = req.ragas_api_key.get_secret_value() if req.ragas_api_key else None

            chunker = AutoChunker(
                mode=req.mode,
                eval_config=EvalConfig(metrics=req.metrics, objective=req.objective),
                proxy_config=ProxyConfig(enabled=req.proxy_enabled, proxy_percent=req.proxy_percent),
                # Ragas Config
                ragas_config=RagasConfig(
                    enabled=req.analyze_ragas,
                    llm_provider=req.ragas_llm_provider,
                    llm_model=req.ragas_llm_model,
                    api_key=ragas_key
                ),
                embedding_provider=req.embedding_provider,
                embedding_model_or_path=req.embedding_model_or_path,
                embedding_api_key=emb_key,
                network_config=NetworkConfig(
                    local_models_path=req.local_models_path
                ),
                telemetry_enabled=req.telemetry_enabled
            )
            
            # Actually run the optimization
            # The optimize call now handles step transitions via on_progress callbacks
            def on_step_progress(msg, step=None):
                if step: 
                    if jobs_data[job_id]["step"] != step:
                        logger.info(f">>> Moving to stage {step}: {msg}")
                    jobs_data[job_id]["step"] = step
                
                # Capture total candidates for progress tracking
                if "parallel evaluation of" in msg.lower():
                    import re
                    match = re.search(r"evaluation of (\d+) candidates", msg)
                    if match:
                        jobs_data[job_id]["total_candidates"] = int(match.group(1))
                        
                add_log(job_id, msg)

            # Handler for real-time result streaming
            def on_result_handler(result_entry):
                # Ensure thread safety if needed (though jobs_data access is generally atomic for dict/list)
                if "partial_results" not in jobs_data[job_id]:
                    jobs_data[job_id]["partial_results"] = []
                jobs_data[job_id]["partial_results"].append(result_entry)
                logger.info(f"Streamed result for {result_entry['name']} to UI (Total: {len(jobs_data[job_id]['partial_results'])})")

            best_plan, report = chunker.optimize(
                documents=req.documents_path,
                candidate_names=req.selected_candidates,
                sweep_params=req.sweep_params,
                on_progress=on_step_progress,
                on_result=on_result_handler  # <--- Pass the new handler
            )

            # Step 4: Persist the winning plan for "Deploy as Code"
            from autochunk.storage.plan import Plan
            plan_path = "best_plan.yaml"
            Plan.write(plan_path, best_plan)
            
            jobs_data[job_id]["result"] = {
                "best_strategy": report["selected"]["name"],
                "best_params": report["selected"]["params"],
                "metrics": report["selected"]["metrics"],
                "candidates": report["candidates"],
                "plan_path": os.path.abspath(plan_path)
            }
            add_log(job_id, f"Optimization complete! Winning plan saved to {plan_path}", is_success=True)
            jobs_data[job_id]["status"] = "completed"
        except asyncio.CancelledError:
            # Client disconnected or task was cancelled
            add_log(job_id, "Optimization cancelled (client disconnected or timeout)")
            jobs_data[job_id]["status"] = "cancelled"
            jobs_data[job_id]["result"] = {"error": "Task cancelled by client or timeout"}
        except Exception as e:
            add_log(job_id, f"Error: {str(e)}")
            jobs_data[job_id]["status"] = "failed"
            jobs_data[job_id]["result"] = {"error": str(e)}

@app.get("/api/docs/list")
def list_documents(path: str):
    if not os.path.exists(path):
        return {"files": [], "error": "Path not found"}
    
    from autochunk.utils.io import SUPPORTED_EXTS
    files = []
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SUPPORTED_EXTS:
                # Use relative path if possible for cleaner UI
                rel_path = os.path.relpath(os.path.join(dirpath, fn), path)
                files.append(rel_path)
    
    return {"files": sorted(files)}

@app.post("/api/optimize")
def start_optimization(req: OptimizeRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{len(jobs_data) + 1}"
    jobs_data[job_id] = {
        "status": "running",
        "logs": [{"time": time.strftime("%H:%M:%S"), "msg": f"Optimization job {job_id} queued", "success": True}],
        "partial_results": [], # <--- Initialize empty list
        "step": 1,
        "result": {}
    }
    background_tasks.add_task(run_optimization_sync, job_id, req)
    return {"job_id": job_id}

@app.get("/api/job/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs_data:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_data[job_id]

# Serve static files
os.makedirs("autochunk/web/static", exist_ok=True)
app.mount("/", StaticFiles(directory="autochunk/web/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
