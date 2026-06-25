from dataclasses import dataclass


@dataclass(kw_only=True)
class DatabaseConfig:
    num_islands: int = 4
    cell_bins: int = 10
    top_k: int = 16
    num_inspirations: int = 3
    migration_size: int = 8
    migration_every: int = 25
