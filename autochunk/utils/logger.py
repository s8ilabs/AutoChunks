import sys, time, logging
from loguru import logger
from contextvars import ContextVar
from typing import List, Dict, Any, Optional

# Global storage for jobs
jobs_data = {}
# ContextVar to track the job_id in the current task/thread
current_job_id: ContextVar[Optional[str]] = ContextVar("current_job_id", default=None)

# Add a Pretty Console Handler (Keep it)
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Custom sink to relay logs to the UI jobs_data
def ui_log_sink(message):
    record = message.record
    # Try extra first (bind), then contextvar
    job_id = record["extra"].get("job_id") or current_job_id.get()
    
    if job_id and job_id in jobs_data:
        msg_text = record["message"]
        level = record["level"].name
        success = level not in ["ERROR", "CRITICAL"]
        
        # Format the message for the UI with its level
        formatted_msg = f"[{level}] {msg_text}"
        
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "msg": formatted_msg,
            "success": success
        }
        jobs_data[job_id]["logs"].append(entry)

# Redirect standard library logging to Loguru
class PropagateHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_library_interception():
    logging.basicConfig(handlers=[PropagateHandler()], level=0, force=True)
    # Target specific verbose libraries if needed
    for name in ["pymupdf", "fitz", "pymupdf4llm"]:
        l = logging.getLogger(name)
        l.handlers = [PropagateHandler()]
        l.propagate = False

setup_library_interception()
logger.add(ui_log_sink, level="INFO")
