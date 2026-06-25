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

    def select_model(self):
        return random.choices(self.llms, weights=[m.cfg.weight for m in self.llms], k=1)[0]

    def generate(self, prompt: str) -> str:
        return self.select_model().generate(prompt)
