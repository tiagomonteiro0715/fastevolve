from ollama import Client, ResponseError

from ..telemetry import log
from .base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model_config, *, host: str = "http://localhost:11434", timeout: float = 600.0, system_prompt: str | None = None):
        self.cfg = model_config
        self.system_prompt = system_prompt
        self.client = Client(host=host, timeout=timeout)
        self._ensure_model()

    def _ensure_model(self):
        try:
            self.client.show(self.cfg.name)
        except ResponseError:
            log.info("[ollama] pulling [bold]%s[/]...", self.cfg.name)
            self.client.pull(self.cfg.name)

    def generate(self, prompt: str) -> str:
        try:
            return self._generate(prompt)
        except Exception:
            log.exception("ollama generate failed for model=%s", self.cfg.name)
            raise

    def _options(self) -> dict:
        opts = {
            "temperature": self.cfg.temperature,
            "num_ctx": self.cfg.num_ctx,
            "flash_attn": self.cfg.flash_attention,
        }
        opts.update(self.cfg.options)
        return opts

    def _generate(self, prompt: str) -> str:
        resp = self.client.generate(
            model=self.cfg.name,
            prompt=prompt,
            system=self.system_prompt,
            options=self._options(),
            stream=False,
        )
        return resp.response
