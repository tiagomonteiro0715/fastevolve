import itertools
from dataclasses import dataclass, field
from typing import Optional, Tuple

_ids = itertools.count()


@dataclass
class Program:
    code: str
    id: int = field(default_factory=lambda: next(_ids), kw_only=True)
    parent_id: Optional[int] = field(default=None, kw_only=True)
    fitness: float = field(default=0.0, kw_only=True)
    behavior: Tuple = field(default=(), kw_only=True)
    island: int = field(default=0, kw_only=True)
    generation: int = field(default=0, kw_only=True)
