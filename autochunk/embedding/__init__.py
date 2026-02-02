
from .base import BaseEncoder
from .local import LocalEncoder
from .hashing import HashingEmbedding
from .openai import OpenAIEncoder
from .ollama import OllamaEncoder

def get_encoder(provider: str, model_name: str, **kwargs) -> BaseEncoder:
    """
     Factory for chunking-aware embeddings.
    Easily extendable to TEI, OpenAI, etc.
    """
    if provider == "local":
        return LocalEncoder(model_name_or_path=model_name, **kwargs)
    elif provider == "hashing":
        return HashingEmbedding(dim=kwargs.get("dim", 256))
    elif provider == "openai":
        return OpenAIEncoder(model_name=model_name, **kwargs)
    elif provider == "ollama":
        return OllamaEncoder(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
