from dataclasses import dataclass, field


@dataclass(kw_only=True)
class EvaluationResult:
    scores: dict = field(default_factory=dict)
    passed: bool = False
    behavior: tuple = ()
    extras: dict = field(default_factory=dict)

    @property
    def fitness(self) -> float:
        return sum(self.scores.values()) / max(1, len(self.scores))
