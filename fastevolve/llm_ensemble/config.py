from dataclasses import dataclass, field
from typing import List, Literal


@dataclass(kw_only=True)
class ModelConfig:
    name: str
    temperature: float = 0.7
    weight: float = 1.0
    role: Literal["fast", "deep"] = "fast"
    num_ctx: int = 4096
    flash_attention: bool = True
    options: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class EnsembleConfig:
    models: List[ModelConfig] = field(default_factory=list)
    host: str = "http://localhost:11434"
    timeout: float = 600.0
