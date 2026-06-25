from dataclasses import dataclass, field
from typing import Callable, List, Tuple


@dataclass(kw_only=True)
class EvaluatorConfig:
    cascade: List[Tuple[Callable, float]] = field(default_factory=list)
    timeout: float = 30.0
    parallelism: int = 1
