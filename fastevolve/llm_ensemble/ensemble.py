import random
from .ollama import OllamaLLM


class LLMEnsemble:
    def __init__(self, config, *, system_prompts: dict | None = None):
        self.config = config
        sp = system_prompts or {}
        self.llms = [OllamaLLM(m, host=config.host, timeout=config.timeout,
                               system_prompt=sp.get(m.name)) for m in config.models]
        if not self.llms:
            raise ValueError("EnsembleConfig.models is empty")

    def select_model(self) -> OllamaLLM:
        return random.choices(self.llms, weights=[m.cfg.weight for m in self.llms], k=1)[0]

    def generate(self, prompt: str) -> str:
        return self.select_model().generate(prompt)
