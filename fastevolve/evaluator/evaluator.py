from queue import PriorityQueue

from ..telemetry import log
from .result import EvaluationResult


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.pool: PriorityQueue = PriorityQueue()

    def execute(self, child_program) -> EvaluationResult:
        result = EvaluationResult()
        for stage_fn, threshold in self.config.cascade:
            name = getattr(stage_fn, "__name__", f"stage{len(result.scores)}")
            try:
                score = float(stage_fn(child_program))
            except Exception:
                log.exception("evaluator stage [yellow]%s[/] raised on program %s", name, child_program.id)
                return result
            result.scores[name] = score
            if score < threshold:
                return result
        result.passed = True
        result.behavior = tuple(max(0.0, min(1.0, v)) for v in result.scores.values())
        return result
