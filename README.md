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
!apt-get -qq install -y zstd
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
