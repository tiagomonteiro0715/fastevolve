import os
import platform
import shutil
import subprocess
import time
from functools import cache

from ollama import Client, ResponseError

from ..exceptions import LLMError
from ..telemetry import log
from .base import BaseLLM


@cache
def _gpu_available() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=2).returncode == 0
    except Exception:
        return False


@cache
def _hardware_info() -> str:
    if _gpu_available():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2,
            )
            gpus = [g.strip() for g in r.stdout.strip().splitlines() if g.strip()]
            if gpus:
                tag = f"{gpus[0]}" + (f" ×{len(gpus)}" if len(gpus) > 1 else "")
                return f"GPU: [bold green]{tag}[/]"
        except Exception:
            pass
        return "GPU: [bold green]detected[/]"
    cpu = platform.processor() or ""
    if not cpu and os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    return f"CPU: [bold yellow]{cpu or 'unknown'}[/] ({os.cpu_count() or '?'} cores)"


def start_ollama(host: str = "127.0.0.1:11434", *, wait: float = 5.0) -> None:
    """Start an ollama daemon with GPU-aware optimizations. No-op if one is already running."""
    for prefix in ("http://", "https://"):
        if host.startswith(prefix):
            host = host[len(prefix):]
    try:
        Client(host=f"http://{host}").list()
        log.info("[ollama] server already running on %s", host)
        return
    except Exception:
        pass

    env = os.environ.copy()
    env["OLLAMA_HOST"] = host
    if _gpu_available():
        env.setdefault("OLLAMA_FLASH_ATTENTION", "1")
        env.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")
        env.setdefault("OLLAMA_NUM_PARALLEL", "4")
        env.setdefault("OLLAMA_MAX_LOADED_MODELS", "2")
        log.info("[ollama] starting server | %s | flash_attn, q8_0 kv-cache, parallel=4, max_loaded=2", _hardware_info())
    else:
        log.info("[ollama] starting server | %s", _hardware_info())

    path = shutil.which("ollama") or "/usr/local/bin/ollama"
    subprocess.Popen([path, "serve"], env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(wait)


class OllamaLLM(BaseLLM):
    def __init__(self, model_config, *, host: str = "http://localhost:11434", timeout: float = 600.0, system_prompt: str | None = None):
        self.cfg = model_config
        self.system_prompt = system_prompt
        self.client = Client(host=host, timeout=timeout)
        self._gpu = _gpu_available()
        try:
            self.client.list()
        except Exception:
            start_ollama(host=host)
            self.client = Client(host=host, timeout=timeout)
        log.info("[ollama] %s | %s", self.cfg.name, _hardware_info())
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
        except Exception as e:
            log.exception("ollama generate failed for model=%s", self.cfg.name)
            raise LLMError(f"ollama:{self.cfg.name}") from e

    def _options(self) -> dict:
        opts = {
            "temperature": self.cfg.temperature,
            "num_ctx": self.cfg.num_ctx,
            "num_predict": self.cfg.max_tokens,
            "flash_attn": self.cfg.flash_attention and self._gpu,
            "num_gpu": -1 if self._gpu else 0,
            "num_thread": 0 if self._gpu else (os.cpu_count() or 4),
        }
        opts.update(self.cfg.options)
        return opts

    def _generate(self, prompt: str) -> str:
        resp = self.client.generate(
            model=self.cfg.name,
            prompt=prompt,
            system=self.system_prompt,
            options=self._options(),
            keep_alive="1h",
            stream=False,
        )
        return resp.response
