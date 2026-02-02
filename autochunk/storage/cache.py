
import os, json
from . import plan as plan_mod
from ..utils.hashing import sha256_hex

class Cache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def path_for(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def get_json(self, key: str):
        p = self.path_for(key + ".json")
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def set_json(self, key: str, value):
        p = self.path_for(key + ".json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

    def put_bytes(self, key: str, data: bytes):
        p = self.path_for(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as f:
            f.write(data)

    def has(self, key: str) -> bool:
        return os.path.exists(self.path_for(key))
