
import re
from typing import List, Callable, Optional
from functools import lru_cache

# ============================================================================
# TOKENIZATION LAYER -  Multi-Backend Support
# ============================================================================

# Global tokenizer cache for performance
_tokenizer_cache = {}

def _get_tiktoken_encoder(model: str = "cl100k_base"):
    """Get or create a tiktoken encoder (cached)."""
    if model not in _tokenizer_cache:
        try:
            import tiktoken
            _tokenizer_cache[model] = tiktoken.get_encoding(model)
        except ImportError:
            return None
    return _tokenizer_cache.get(model)

def get_tokens(text: str, tokenizer: str = "auto") -> List[str]:
    """
    Unified tokenization across all chunkers.
    
    Args:
        text: Input text to tokenize
        tokenizer: One of "auto", "tiktoken", "whitespace", "character"
                   "auto" tries tiktoken first, falls back to whitespace
    
    Returns:
        List of token strings
    """
    if tokenizer == "auto":
        # Try tiktoken first (most accurate for GPT models)
        enc = _get_tiktoken_encoder()
        if enc:
            token_ids = enc.encode(text)
            # Return token strings for compatibility
            return [enc.decode([t]) for t in token_ids]
        # Fallback to whitespace
        tokenizer = "whitespace"
    
    if tokenizer == "tiktoken":
        enc = _get_tiktoken_encoder()
        if enc:
            token_ids = enc.encode(text)
            return [enc.decode([t]) for t in token_ids]
        raise ImportError("tiktoken not installed. Run: pip install tiktoken")
    
    if tokenizer == "character":
        return list(text)
    
    # Default: whitespace + punctuation splitting
    return [t for t in re.split(r"(\s+|[.,!?;:'\"\(\)\[\]\{\}])", text) if t]

def decode_tokens(tokens: List[str]) -> str:
    """Unified detokenization."""
    return "".join(tokens)

def count_tokens(text: str, tokenizer: str = "auto") -> int:
    """
    Count tokens using the specified tokenizer.
    
    For production RAG with GPT models, use tokenizer="tiktoken".
    """
    if not text:
        return 0
    
    if tokenizer == "auto" or tokenizer == "tiktoken":
        enc = _get_tiktoken_encoder()
        if enc:
            return len(enc.encode(text))
    
    if tokenizer == "character":
        return len(text)
    
    # Fallback to whitespace tokenization
    return len([t for t in re.split(r"(\s+|[.,!?;:'\"\(\)\[\]\{\}])", text) if t.strip()])

def create_length_function(method: str = "token") -> Callable[[str], int]:
    """
    Factory for length functions (LangChain compatibility).
    
    Args:
        method: "token" (tiktoken), "char" (character count), "word" (word count)
    
    Returns:
        A function that takes text and returns length
    """
    if method == "token":
        return lambda text: count_tokens(text, tokenizer="auto")
    elif method == "char":
        return len
    elif method == "word":
        return lambda text: len(text.split())
    else:
        return lambda text: count_tokens(text, tokenizer="auto")

# ============================================================================
# SENTENCE SPLITTING -  Multi-Backend Support
# ============================================================================

def split_sentences(text: str, backend: str = "auto") -> List[str]:
    """
     Sentence Splitting with multiple backends.
    
    Args:
        text: Input text
        backend: "auto", "nltk", "spacy", "regex"
    
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    if backend == "auto" or backend == "nltk":
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt_tab', quiet=True)
            return nltk.sent_tokenize(text)
        except Exception:
            if backend == "nltk":
                raise
            # Fall through to regex
    
    if backend == "spacy":
        try:
            import spacy
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        except ImportError:
            raise ImportError("spacy not installed. Run: pip install spacy")
    
    # Regex fallback - handles abbreviations better than naive split
    # Negative lookbehinds for common abbreviations
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=[.!?])\s+'
    parts = re.split(pattern, text.strip())
    return [p.strip() for p in parts if p.strip()]

def whitespace_tokens(text: str) -> List[str]:
    """Keep whitespace tokens for reconstruction."""
    return [t for t in re.split(r"(\s+)", text) if t]

# ============================================================================
# TEXT UTILITIES
# ============================================================================

def get_char_to_token_map(text: str) -> List[int]:
    """
    Create a mapping from character index to token index.
    Useful for tracking start indices in chunks.
    """
    tokens = get_tokens(text)
    char_map = []
    token_idx = 0
    char_pos = 0
    
    for i, token in enumerate(tokens):
        token_len = len(token)
        for _ in range(token_len):
            char_map.append(i)
        char_pos += token_len
    
    return char_map

def extract_code_blocks(text: str) -> List[dict]:
    """
    Extract fenced code blocks from markdown text.
    Returns list of {"start": int, "end": int, "content": str, "language": str}
    """
    pattern = r'```(\w*)\n(.*?)```'
    blocks = []
    for match in re.finditer(pattern, text, re.DOTALL):
        blocks.append({
            "start": match.start(),
            "end": match.end(),
            "content": match.group(2),
            "language": match.group(1) or "text"
        })
    return blocks

def is_inside_code_block(text: str, position: int) -> bool:
    """Check if a character position is inside a fenced code block."""
    blocks = extract_code_blocks(text)
    for block in blocks:
        if block["start"] <= position < block["end"]:
            return True
    return False
