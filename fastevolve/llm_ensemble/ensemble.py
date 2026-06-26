import random


def _make_llm(mcfg, **kw):
    if mcfg.provider == "ollama":
        from .ollama import OllamaLLM
        return OllamaLLM(mcfg, **kw)
    if mcfg.provider == "openai":
        from .openai_llm import OpenAILLM
        return OpenAILLM(mcfg, **kw)
    if mcfg.provider == "anthropic":
        from .anthropic_llm import ClaudeLLM
        return ClaudeLLM(mcfg, **kw)
    raise ValueError(f"unknown provider: {mcfg.provider}")


class LLMEnsemble:
    def __init__(self, config, *, system_prompts: dict | None = None):
        self.config = config
        sp = system_prompts or {}
        self.llms = [_make_llm(m, host=config.host, timeout=config.timeout,
                               system_prompt=sp.get(m.name)) for m in config.models]
        if not self.llms:
            raise ValueError("EnsembleConfig.models is empty")
        self._stats = [[1, 1] for _ in self.llms]  # [calls, wins] with Laplace prior
        self._last = 0
        self.epsilon = 0.1

    def select_model(self):
        if random.random() < self.epsilon or any(s[0] < 5 for s in self._stats):
            self._last = random.choices(
                range(len(self.llms)), weights=[m.cfg.weight for m in self.llms], k=1
            )[0]
        else:
            self._last = max(range(len(self.llms)),
                             key=lambda i: self._stats[i][1] / self._stats[i][0])
        return self.llms[self._last]

    def generate(self, prompt: str) -> str:
        return self.select_model().generate(prompt)

    def feedback(self, success: bool) -> None:
        self._stats[self._last][0] += 1
        if success:
            self._stats[self._last][1] += 1
