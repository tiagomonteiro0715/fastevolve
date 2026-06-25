from openai import OpenAI

from ..telemetry import log
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model_config, *, system_prompt: str | None = None, **_):
        self.cfg = model_config
        self.system_prompt = system_prompt
        self.client = OpenAI()  # reads OPENAI_API_KEY

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
                **self.cfg.options,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            log.exception("openai generate failed for model=%s", self.cfg.name)
            raise
