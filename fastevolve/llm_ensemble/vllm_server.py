"""vLLM OpenAI-compatible server launcher.

CUDA LIMITATION: vLLM ships pre-built wheels tied to specific CUDA versions.
`fastevolve[vllm]` installs the PyPI default (currently CUDA 12.1). It is
impossible to make a single pin work across all GPUs / CUDA versions. If your
driver doesn't match, install vLLM yourself BEFORE adding the extra:

    pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
    pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124

then `uv add 'fastevolve[vllm]'` is a no-op verification step.
"""
import socket
import subprocess
import sys
import time
import urllib.request
from functools import cache

from ..telemetry import log
from .ollama import _hardware_info


@cache
def _vllm_info() -> dict:
    info = {"vllm_ver": None, "driver_cuda": "?"}
    try:
        import vllm
        info["vllm_ver"] = vllm.__version__
    except ImportError:
        return info
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=2)
        for ln in r.stdout.splitlines():
            if "CUDA Version:" in ln:
                info["driver_cuda"] = ln.split("CUDA Version:")[1].split()[0]
                break
    except Exception:
        pass
    return info


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except Exception:
        return False


def start_vllm(model: str, *, host: str = "127.0.0.1", port: int = 8000,
               wait: float = 180.0, verbose: bool = False, **vllm_kwargs) -> None:
    """Spawn a vLLM OpenAI-compatible server in the background. Idempotent.

    Logs the detected vLLM version, driver CUDA, and GPU on startup so any
    CUDA-mismatch surfaces immediately. See module docstring for the
    CUDA-version limitation.

    Set `verbose=True` to stream vLLM's own logs (weight download progress,
    model loading, ready message). Useful for first-run model downloads,
    which can take 5–30 minutes for large models. Default is quiet — we
    still log a periodic "still initializing" heartbeat every 10s.
    """
    info = _vllm_info()
    if info["vllm_ver"] is None:
        raise RuntimeError(
            "vLLM is not installed. Try:  uv add 'fastevolve[vllm]'\n"
            "If your CUDA isn't 12.1, install vLLM manually with the right "
            "--extra-index-url first (see fastevolve.llm_ensemble.vllm_server docstring)."
        )
    if _port_open(host, port):
        log.info("[vllm] server already running on %s:%d", host, port)
        return
    log.info("[vllm] starting | vLLM %s | driver CUDA %s | %s",
             info["vllm_ver"], info["driver_cuda"], _hardware_info())

    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--host", host, "--port", str(port)]
    for k, v in vllm_kwargs.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    out = None if verbose else subprocess.DEVNULL
    subprocess.Popen(cmd, stdout=out, stderr=out)

    started = time.time()
    deadline = started + wait
    last_log = started
    url = f"http://{host}:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    log.info("[vllm] [bold green]ready[/] at %s (took %.0fs)", url, time.time() - started)
                    return
        except Exception:
            now = time.time()
            if now - last_log >= 10:
                log.info("[vllm] still initializing... [yellow]%.0fs elapsed[/] (set verbose=True for vLLM's own logs)", now - started)
                last_log = now
            time.sleep(2)
    log.warning("[vllm] server did not become ready in %.0fs — proceeding", wait)
