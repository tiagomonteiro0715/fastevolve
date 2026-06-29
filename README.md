# fastevolve

Minimal open-source AlphaEvolve: LLM-driven program evolution with MAP-Elites islands, cascade evaluation, and a local Ollama ensemble.

## Install

### 1. Install uv (one-time)

uv is a fast Python package manager. Pick the line for your OS:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or via Homebrew (`brew install uv`), pipx (`pipx install uv`), or pip (`pip install uv`).

### 2. Add fastevolve to a new project

```bash
uv init my-evolve-project
cd my-evolve-project
uv add fastevolve
```

OpenAI and Anthropic SDKs are optional extras — install whichever you need:

```bash
uv add "fastevolve[openai]"       # adds the OpenAI SDK
uv add "fastevolve[anthropic]"    # adds the Anthropic SDK
uv add "fastevolve[all]"          # both
```

If you only use Ollama, skip the extras — neither SDK will be imported.

### 3. Or clone this repo and sync

```bash
git clone https://github.com/tiagomonteiro0715/fastevolve.git
cd fastevolve
uv sync                       # core
uv sync --extra all           # core + OpenAI + Anthropic
```

## Quick start in code

### Local (with Ollama)

Assumes `ollama serve` is running and you've pulled the model.

The ensemble below mixes a **fast** model (low temperature, conservative) with a **deep** one (higher temperature, more exploration). The adaptive router learns over time when to call each. Evaluation is wrapped in `run_sandboxed`, which kills any program that loops forever or crashes — that iteration just scores zero instead of hanging the run.

```python
from fastevolve import Config, Controller, run_sandboxed
from fastevolve.llm_ensemble import ModelConfig

INITIAL = "def solve(x):\n    return x\n"

def correctness(p):
    cases = [(2, 4), (3, 9), (4, 16), (5, 25)]
    return sum(1 for x, y in cases
               if run_sandboxed(p.code, "solve", x, timeout=2.0) == y) / len(cases)

cfg = Config()
cfg.iterations = 20
cfg.checkpoint_path = "run.log"   # optional — resume if killed mid-run
cfg.ensemble.models = [
    ModelConfig(name="gemma4:e2b", provider="ollama", temperature=0.4, weight=1.0, role="fast"),
    ModelConfig(name="gemma4:e2b", provider="ollama", temperature=0.9, weight=0.5, role="deep"),
]
cfg.evaluator.cascade = [(correctness, 0.0)]

result = Controller(cfg, initial_program=INITIAL).run()
print(result.best.code)
```

### Google Colab (with OpenAI or Anthropic)

Ollama isn't practical on Colab — use an API provider instead. Paste this into a Colab cell:

```python
!pip install -q "fastevolve[openai]"

import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")  # store in Colab Secrets first

from fastevolve import Config, Controller, run_sandboxed
from fastevolve.llm_ensemble import ModelConfig

INITIAL = "def solve(x):\n    return x\n"

def correctness(p):
    cases = [(2, 4), (3, 9), (4, 16), (5, 25)]
    return sum(1 for x, y in cases
               if run_sandboxed(p.code, "solve", x, timeout=2.0) == y) / len(cases)

cfg = Config()
cfg.iterations = 20
cfg.ensemble.models = [
    ModelConfig(name="gpt-4o-mini", provider="openai", temperature=0.4, weight=1.0, role="fast"),
    ModelConfig(name="gpt-4o",      provider="openai", temperature=0.7, weight=0.3, role="deep"),
]
cfg.evaluator.cascade = [(correctness, 0.0)]

result = Controller(cfg, initial_program=INITIAL).run()
print(result.best.code)
```

### Google Colab (with Claude)

```python
!pip install -q "fastevolve[anthropic]"

import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")  # store in Colab Secrets first

from fastevolve import Config, Controller, run_sandboxed
from fastevolve.llm_ensemble import ModelConfig

INITIAL = "def solve(x):\n    return x\n"

def correctness(p):
    cases = [(2, 4), (3, 9), (4, 16), (5, 25)]
    return sum(1 for x, y in cases
               if run_sandboxed(p.code, "solve", x, timeout=2.0) == y) / len(cases)

cfg = Config()
cfg.iterations = 20
cfg.ensemble.models = [
    ModelConfig(name="claude-haiku-4-5-20251001", provider="anthropic",
                temperature=0.4, weight=1.0, role="fast"),
    ModelConfig(name="claude-opus-4-7",          provider="anthropic",
                temperature=0.7, weight=0.3, role="deep"),
]
cfg.evaluator.cascade = [(correctness, 0.0)]

result = Controller(cfg, initial_program=INITIAL).run()
print(result.best.code)
```

### Google Colab (with Ollama)

Ollama can run on Colab if you install it, start the daemon in the background, and pull a model. Tested working on the free CPU runtime with a tiny model (`qwen2.5:0.5b`).

**On Colab Pro / Pro+**: switch to an A100 or L4 GPU runtime (`Runtime → Change runtime type → A100 GPU`) and swap the model for something bigger — `qwen2.5-coder:7b`, `llama3.1:8b`, or `gemma2:9b` all fit comfortably and produce dramatically better evolution candidates than `0.5b`. Pro+'s longer sessions (24 h) and background execution also mean you can leave a 1000-iteration run going overnight without keeping the tab open.

```python
# 1. Install ollama (zstd is required by the install script) and fastevolve via uv
!apt-get -qq install -y zstd pciutils lshw
!curl -fsSL https://ollama.com/install.sh | sh
!pip install uv
!uv pip install -q fastevolve

# 2. Run fastevolve — it starts the ollama daemon automatically with GPU-aware
#    optimizations (flash attention, q8_0 KV cache, parallel decoding) when a GPU is detected.
from fastevolve import Config, Controller, run_sandboxed
from fastevolve.llm_ensemble import ModelConfig

INITIAL = "def solve(x):\n    return x\n"

def correctness(p):
    cases = [(2, 4), (3, 9), (4, 16), (5, 25)]
    return sum(1 for x, y in cases
               if run_sandboxed(p.code, "solve", x, timeout=2.0) == y) / len(cases)

cfg = Config()
cfg.iterations = 20
cfg.ensemble.models = [
    # free CPU runtime: only the small fast model
    ModelConfig(name="qwen2.5:0.5b",       provider="ollama",
                temperature=0.4, weight=1.0, role="fast"),
    # Pro / Pro+ A100 or L4: add a stronger deep model — the router will escalate when needed
    ModelConfig(name="qwen2.5-coder:7b",   provider="ollama",
                temperature=0.7, weight=0.5, role="deep"),
]
cfg.evaluator.cascade = [(correctness, 0.0)]

result = Controller(cfg, initial_program=INITIAL).run()
print(result.best.code)
```

Colab sessions are disconnected after ~90 min idle and the VM is wiped — set `cfg.checkpoint_path = "/content/drive/MyDrive/run.log"` after mounting Drive if you want resume across sessions.

### Google Colab (with vLLM)

vLLM is a high-throughput inference engine with continuous batching and paged attention — significantly faster than Ollama for sustained generation and multi-GPU setups. Use it when you have an A100 / L4 / H100 and care about throughput.

**CUDA limitation (important).** vLLM ships pre-built wheels tied to specific CUDA versions and it is **impossible to make a single pin work for every GPU**. `fastevolve[vllm]` installs the PyPI default, which targets **CUDA 12.1** — this works on Colab Pro+ A100/L4 runtimes (currently CUDA 12.x) and most modern cloud GPUs. If your driver is on a different CUDA version, install vLLM yourself first with the matching wheel:

```bash
# Pick the line that matches your driver's CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118    # CUDA 11.8
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124    # CUDA 12.4
# then:
uv add "fastevolve[vllm]"
```

vLLM is Linux-only (or Linux via WSL2). It does not work on Windows or macOS natively.

The library will log the detected vLLM version, driver CUDA, and GPU at startup so any mismatch surfaces immediately.

The ensemble below runs **two different vLLM servers on the same GPU**: a small, fast Qwen for cheap exploration and a bigger Gemma for hard cases. The adaptive router learns when to escalate. Each vLLM server is told to use ~45 % of GPU memory so they coexist on a single A100 40 GB.

```python
# 1. Pick an A100 / L4 / H100 GPU runtime: Runtime → Change runtime type → A100 GPU
# 2. Install fastevolve with the vLLM extra (works on Colab's CUDA 12.x out of the box)
!pip install -q "fastevolve[vllm]"

# 3. Start two vLLM OpenAI-compatible servers on different ports.
#    `gpu_memory_utilization=0.45` lets both fit on one 40 GB A100.
#    `verbose=True` streams vLLM's own logs — useful on the first run because
#    it downloads model weights from Hugging Face (5–30 min for big models).
#    Drop `verbose=True` later to silence the noise once weights are cached.
from fastevolve.llm_ensemble import start_vllm
start_vllm("Qwen/Qwen2.5-Coder-1.5B-Instruct", port=8000,
           gpu_memory_utilization=0.45, verbose=True)
start_vllm("google/gemma-2-9b-it",             port=8001,
           gpu_memory_utilization=0.45, verbose=True)

# 4. Run fastevolve — each ModelConfig points at one of the local servers via base_url
from fastevolve import Config, Controller, run_sandboxed
from fastevolve.llm_ensemble import ModelConfig

# Task: parse a Roman numeral into an integer.
# The seed handles pure addition (e.g. "III" = 3, "LVIII" = 58) but misses the
# subtractive notation (e.g. "IV" = 4, "MCMXCIV" = 1994). The LLM has to spot
# and add the subtraction rule. Harder than `x*x` — needs real reasoning, not
# just curve-fitting to a few input/output pairs.
INITIAL = '''def solve(s):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    return sum(vals[c] for c in s)
'''

CASES = [
    ("III", 3), ("VIII", 8), ("LVIII", 58),         # pure addition — seed already passes
    ("IV", 4), ("IX", 9), ("XL", 40), ("XC", 90),   # simple subtraction
    ("CDXLIV", 444), ("MCMXCIV", 1994),             # nested subtraction
    ("MMXXIV", 2024), ("MMMCMXCIX", 3999),          # long-form
]

def correctness(p):
    return sum(1 for s, y in CASES
               if run_sandboxed(p.code, "solve", s, timeout=2.0) == y) / len(CASES)

cfg = Config()
cfg.iterations = 50
cfg.ensemble.models = [
    # fast: small Qwen, low temperature, high weight — dominates the cheap iterations
    ModelConfig(name="Qwen/Qwen2.5-Coder-1.5B-Instruct", provider="openai",
                base_url="http://127.0.0.1:8000/v1",
                temperature=0.4, weight=1.0, role="fast"),
    # deep: bigger Gemma, higher temperature, lower weight — escalated by the router when fast stalls
    ModelConfig(name="google/gemma-2-9b-it", provider="openai",
                base_url="http://127.0.0.1:8001/v1",
                temperature=0.8, weight=0.3, role="deep"),
]
cfg.evaluator.cascade = [(correctness, 0.0)]

result = Controller(cfg, initial_program=INITIAL).run()
print(result.best.code)
```

You can mix any combination — a coding-specialized model as fast + a general reasoning model as deep, a small distilled model + a larger base model, etc. The router only cares which `ModelConfig` produces fitness improvements; it doesn't care about the architecture or vendor behind each `base_url`.

For multi-GPU runtimes (two A100s, etc.), pass tensor-parallel size to put a single large model across both GPUs:

```python
start_vllm("Qwen/Qwen2.5-Coder-32B-Instruct", tensor_parallel_size=2)
```

### `base_url` works with any OpenAI-compatible server (not just vLLM)

The `base_url` field on `ModelConfig` isn't vLLM-specific — it's how you point fastevolve at any server that speaks the OpenAI Chat Completions API. That means you can self-host with whichever engine you prefer and the library treats it as just another model in the ensemble:

| Engine | Typical `base_url` | Notes |
|---|---|---|
| vLLM | `http://127.0.0.1:8000/v1` | Highest throughput on NVIDIA GPUs; what `start_vllm()` spawns. |
| LM Studio | `http://localhost:1234/v1` | GUI desktop app, easy on macOS/Windows. |
| llama.cpp server | `http://localhost:8080/v1` | CPU-friendly, GGUF quantization, runs anywhere. |
| TGI (Hugging Face Text Generation Inference) | `http://localhost:8080/v1` | Production-grade serving, similar throughput to vLLM. |
| SGLang | `http://localhost:30000/v1` | Optimized for structured generation and constrained decoding. |
| Any OpenAI-compatible cloud endpoint | their published URL | Use the provider's `api_key` via the `OPENAI_API_KEY` env var. |

You start the server however that engine's docs say to, then drop a `ModelConfig` into the ensemble:

```python
ModelConfig(name="my-served-model", provider="openai",
            base_url="http://localhost:1234/v1",
            temperature=0.5, weight=1.0, role="fast")
```

Mix and match freely — a llama.cpp CPU server for cheap iterations + a TGI GPU server for the deep model, for example. The adaptive router doesn't care which engine is behind each URL; it only cares which one produces fitness improvements.

## Run the demo

Start Ollama and pull the model first:

```bash
ollama serve
ollama pull gemma4:e4b
```

Then:

```bash
uv run python main.py
```

## Using OpenAI or Claude in the ensemble

Set the API key for whichever provider(s) you plan to use:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

On Windows (cmd.exe): `set OPENAI_API_KEY=sk-...`

Then pick a `provider` per model in your config. You can freely mix providers in one ensemble:

```python
from fastevolve.llm_ensemble import ModelConfig

cfg.ensemble.models = [
    ModelConfig(name="gemma4:e4b", provider="ollama", temperature=0.6, weight=1.0, role="fast"),
    ModelConfig(name="gpt-4o-mini", provider="openai", temperature=0.6, weight=1.0, role="fast"),
    ModelConfig(name="claude-opus-4-7", provider="anthropic", temperature=0.7, weight=1.0,
                role="deep", options={"max_tokens": 4096}),
]
```

`provider` defaults to `"ollama"`, so existing configs keep working unchanged.
