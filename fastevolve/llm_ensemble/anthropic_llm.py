import httpx
from anthropic import Anthropic

from ..exceptions import LLMError
from ..telemetry import log
from .base import BaseLLM

_HTTP = httpx.Client(limits=httpx.Limits(max_keepalive_connections=10, max_connections=20))


class ClaudeLLM(BaseLLM):
    def __init__(self, model_config, *, system_prompt: str | None = None, **_):
        self.cfg = model_config
        self.system_prompt = system_prompt
        self.client = Anthropic(http_client=_HTTP)

    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.messages.create(
                model=self.cfg.name,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                system=self.system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            log.exception("anthropic generate failed for model=%s", self.cfg.name)
            raise LLMError(f"anthropic:{self.cfg.name}") from e
