
from dataclasses import dataclass
from typing import Callable

@dataclass
class SimpleTokenizer:
    name: str = "whitespace"
    def tokens(self, text: str):
        return text.split()
