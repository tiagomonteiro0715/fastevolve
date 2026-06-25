from dataclasses import dataclass, field
from .prompt_sampler.config import PromptConfig
from .llm_ensemble.config import EnsembleConfig
from .evaluator.config import EvaluatorConfig
from .program_database.config import DatabaseConfig


@dataclass(kw_only=True)
class Config:
    prompt: PromptConfig = field(default_factory=PromptConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    iterations: int = 100
