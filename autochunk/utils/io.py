import os, hashlib
from typing import List, Dict, Optional, Callable
from .logger import logger
from .hashing import sha256_hex

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".html", ".htm"}

def read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def _load_pdf_markdown(path: str, on_progress: Optional[Callable[[str], None]] = None) -> str:
    """Progressive  PDF to Markdown extraction for AutoChunks native chunkers."""
    import fitz
    doc = fitz.open(path)
    total_pages = len(doc)
    
    if on_progress: 
        on_progress(f"Analyzing structure of {total_pages} pages...")
        logger.info(f"Deep Analysis started for {os.path.basename(path)} ({total_pages} pages)")

    full_text = []
    
    try:
        import pymupdf4llm
        # Process in chunks of 5 pages for high-frequency real-time progress
        chunk_size = 5
        for i in range(0, total_pages, chunk_size):
            end_page = min(i + chunk_size, total_pages)
            if on_progress: 
                on_progress(f"Extracting layout (Pages {i+1}-{end_page} of {total_pages})...")
            
            # Extract specific page range
            page_text = pymupdf4llm.to_markdown(path, pages=list(range(i, end_page)))
            full_text.append(page_text)
            
        return "\n".join(full_text)
        
    except Exception as e:
        logger.warning(f" extraction failed, falling back to basic text: {e}")
        return _load_pdf_raw(path, on_progress)

def _load_pdf_raw(path: str, on_progress: Optional[Callable[[str], None]] = None) -> str:
    """Raw text extraction from PDF for fair bridge comparison."""
    import fitz
    doc = fitz.open(path)
    total_pages = len(doc)
    if on_progress: on_progress(f"Fast extraction: {total_pages} pages...")
    return "\n".join([page.get_text() for page in doc])

def _load_html_processed(path: str) -> str:
    """Clean HTML to plain text using BeautifulSoup - for AutoChunks native chunkers."""
    try:
        from bs4 import BeautifulSoup
        content = read_text_file(path)
        soup = BeautifulSoup(content, "lxml")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text(separator=' ')
        
        # Break into lines and remove leading and trailing whitespace
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        logger.warning(f"HTML parsing failed for {path}: {e}")
        return read_text_file(path)

def _load_html_raw(path: str) -> str:
    """Raw HTML text for LangChain HTML splitter."""
    return read_text_file(path)

def load_documents(root: str, on_progress: Optional[Callable[[str], None]] = None, high_fidelity: bool = True) -> List[Dict]:
    """
    Load documents with BOTH raw and processed versions for fair evaluation.
    
    Returns documents with:
    - text: Processed version (for AutoChunks native chunkers)
    - raw_text: Original/raw version (for LangChain bridges)
    """
    if on_progress: on_progress(f"Scanning directory: {root}")
    docs = []
    
    # Text cache directory
    cache_root = os.path.join(os.getcwd(), ".autochunk", "text_cache")
    os.makedirs(cache_root, exist_ok=True)
    
    if not os.path.exists(root):
        raise FileNotFoundError(f"Path not found: {root}")
        
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXTS:
                p = os.path.abspath(os.path.normpath(os.path.join(dirpath, fn)))
                try:
                    # 1. Quick Binary Hash
                    with open(p, "rb") as bf:
                        binary_data = bf.read()
                        b_hash = sha256_hex(binary_data)
                    
                    cache_file_proc = os.path.join(cache_root, f"{b_hash}_processed.txt")
                    cache_file_raw = os.path.join(cache_root, f"{b_hash}_raw.txt")
                    
                    text_processed = ""
                    text_raw = ""
                    
                    # Check cache for processed text
                    if os.path.exists(cache_file_proc) and os.path.exists(cache_file_raw):
                        # Cache Hit!
                        if on_progress: on_progress(f"⚡ Cache Hit: Loading {fn}...")
                        logger.info(f"CACHE HIT: Reusing text for {fn}")
                        with open(cache_file_proc, "r", encoding="utf-8") as tf:
                            text_processed = tf.read()
                        with open(cache_file_raw, "r", encoding="utf-8") as tf:
                            text_raw = tf.read()
                    else:
                        # Cache Miss: Load both versions
                        if on_progress: on_progress(f"Processing ({len(docs)+1}): {fn}...")
                        logger.info(f"Loading document: {fn}")
                        
                        if ext == ".pdf":
                            # Processed: Markdown extraction (AutoChunks advantage)
                            if high_fidelity:
                                text_processed = _load_pdf_markdown(p, on_progress)
                            else:
                                text_processed = _load_pdf_raw(p, on_progress)
                            # Raw: Basic text extraction (fair for bridges)
                            text_raw = _load_pdf_raw(p, None)
                            
                        elif ext in [".html", ".htm"]:
                            # Processed: BeautifulSoup cleaned (for text-based chunkers)
                            text_processed = _load_html_processed(p)
                            # Raw: Original HTML (for LangChain HTML splitter)
                            text_raw = _load_html_raw(p)
                            
                        else:
                            # Plain text files: same for both
                            text_processed = read_text_file(p)
                            text_raw = text_processed
                        
                        # Save to cache
                        if text_processed.strip():
                            with open(cache_file_proc, "w", encoding="utf-8") as tf:
                                tf.write(text_processed)
                        if text_raw.strip():
                            with open(cache_file_raw, "w", encoding="utf-8") as tf:
                                tf.write(text_raw)
                    
                    if text_processed.strip() or text_raw.strip():
                        docs.append({
                            "id": p, 
                            "path": p, 
                            "text": text_processed,      # For AutoChunks native chunkers
                            "raw_text": text_raw,        # For LangChain bridges
                            "ext": ext,
                            "hash": b_hash
                        })
                        logger.success(f"Loaded {fn} (processed: {len(text_processed)} chars, raw: {len(text_raw)} chars)")
                    else:
                        logger.warning(f"Skipping empty document: {p}")
                except Exception as e:
                    logger.error(f"Failed to load {p}: {e}")
                    
    if not docs:
        raise RuntimeError(f"No supported documents found in {root} (supported: {SUPPORTED_EXTS})")
        
    logger.info(f"Loaded {len(docs)} documents from {root}")
    return docs
