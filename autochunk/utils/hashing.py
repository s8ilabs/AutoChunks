
import hashlib

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def content_hash(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))
