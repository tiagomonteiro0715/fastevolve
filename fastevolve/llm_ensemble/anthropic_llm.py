from anthropic import Anthropic

from ..telemetry import log
from .base import BaseLLM


class ClaudeLLM(BaseLLM):
    def __init__(self, model_config, *, system_prompt: str | None = None, **_):
        self.cfg = model_config
        self.system_prompt = system_prompt
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY

    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.messages.create(
                model=self.cfg.name,
                max_tokens=self.cfg.options.get("max_tokens", 4096),
                temperature=self.cfg.temperature,
                system=self.system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception:
            log.exception("anthropic generate failed for model=%s", self.cfg.name)
            raise
