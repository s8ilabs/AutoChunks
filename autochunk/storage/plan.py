
from __future__ import annotations
import os, json, time
from dataclasses import dataclass, asdict, field
from typing import Dict, Any
import yaml

@dataclass
class Plan:
    id: str
    corpus_hash: str
    generator_pipeline: Dict[str, Any]
    metrics: Dict[str, Any]
    embedding: Dict[str, Any]
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def apply(self, documents, chunker) -> list:
        """Delegate application to chunker with the saved generator params."""
        gen_name = self.generator_pipeline.get("name")
        params = self.generator_pipeline.get("params", {})
        return chunker.apply_with_generator(documents, gen_name, params)

    def to_langchain(self) -> 'AutoChunkLangChainAdapter':
        from ..adapters.langchain import AutoChunkLangChainAdapter
        return AutoChunkLangChainAdapter(plan=self)

    def to_llamaindex(self) -> 'AutoChunkLlamaIndexAdapter':
        from ..adapters.llamaindex import AutoChunkLlamaIndexAdapter
        return AutoChunkLlamaIndexAdapter(plan=self)

    @staticmethod
    def write(path: str, plan: 'Plan'):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(asdict(plan), f, sort_keys=False, allow_unicode=True)

    @staticmethod
    def read(path: str) -> 'Plan':
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return Plan(**data)
