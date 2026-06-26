class FastEvolveError(Exception):
    """Base for all fastevolve errors."""


class EvaluatorError(FastEvolveError):
    """A user-supplied cascade stage raised."""


class LLMError(FastEvolveError):
    """An LLM provider call failed."""


class CheckpointError(FastEvolveError):
    """Checkpoint read/write failed."""
