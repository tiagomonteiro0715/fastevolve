import os

import httpx
from openai import OpenAI

from ..exceptions import LLMError
from ..telemetry import log
from .base import BaseLLM

_HTTP = httpx.Client(limits=httpx.Limits(max_keepalive_connections=10, max_connections=20))


class OpenAILLM(BaseLLM):
    def __init__(self, model_config, *, system_prompt: str | None = None, **_):
        self.cfg = model_config
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
            base_url=self.cfg.base_url,
            http_client=_HTTP,
        )

    def generate(self, prompt: str) -> str:
        try:
            msgs = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs.append({"role": "user", "content": prompt})
            resp = self.client.chat.completions.create(
                model=self.cfg.name,
                messages=msgs,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                **self.cfg.options,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            log.exception("openai generate failed for model=%s", self.cfg.name)
            raise LLMError(f"openai:{self.cfg.name}") from e
