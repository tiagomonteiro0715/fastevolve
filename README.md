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

```python
from fastevolve import Config, Controller
from fastevolve.llm_ensemble import ModelConfig

INITIAL = "def solve(x):\n    return x\n"

def correctness(p):
    ns = {}
    try: exec(p.code, ns)
    except Exception: return 0.0
    fn = ns.get("solve")
    cases = [(2, 4), (3, 9), (4, 16), (5, 25)]
    return sum(1 for x, y in cases if fn and fn(x) == y) / len(cases)

cfg = Config()
cfg.iterations = 20
cfg.checkpoint_path = "run.log"   # optional — resume if killed mid-run
cfg.ensemble.models = [
    ModelConfig(name="gemma3:e4b", provider="ollama", temperature=0.7, weight=1.0, role="fast"),
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

from fastevolve import Config, Controller
from fastevolve.llm_ensemble import ModelConfig

INITIAL = "def solve(x):\n    return x\n"

def correctness(p):
    ns = {}
    try: exec(p.code, ns)
    except Exception: return 0.0
    fn = ns.get("solve")
    cases = [(2, 4), (3, 9), (4, 16), (5, 25)]
    return sum(1 for x, y in cases if fn and fn(x) == y) / len(cases)

cfg = Config()
cfg.iterations = 20
cfg.ensemble.models = [
    ModelConfig(name="gpt-4o-mini", provider="openai", temperature=0.7, weight=1.0, role="fast"),
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

from fastevolve import Config, Controller
from fastevolve.llm_ensemble import ModelConfig

INITIAL = "def solve(x):\n    return x\n"

def correctness(p):
    ns = {}
    try: exec(p.code, ns)
    except Exception: return 0.0
    fn = ns.get("solve")
    cases = [(2, 4), (3, 9), (4, 16), (5, 25)]
    return sum(1 for x, y in cases if fn and fn(x) == y) / len(cases)

cfg = Config()
cfg.iterations = 20
cfg.ensemble.models = [
    ModelConfig(name="claude-haiku-4-5-20251001", provider="anthropic",
                temperature=0.7, weight=1.0, role="fast",
                options={"max_tokens": 4096}),
]
cfg.evaluator.cascade = [(correctness, 0.0)]

result = Controller(cfg, initial_program=INITIAL).run()
print(result.best.code)
```

### Google Colab (with Ollama)

Ollama can run on Colab if you install it, start the daemon in the background, and pull a small model. Use a GPU runtime (`Runtime → Change runtime type → T4 GPU`) for any model bigger than ~1B parameters.

```python
# 1. Install ollama and fastevolve
!curl -fsSL https://ollama.com/install.sh | sh
!pip install -q fastevolve

# 2. Start the ollama daemon in the background
import subprocess, time
subprocess.Popen(["ollama", "serve"])
time.sleep(5)  # give it a moment to bind to port 11434

# 3. Pull a small model (qwen2.5:0.5b is ~400 MB and fits the free CPU runtime)
!ollama pull qwen2.5:0.5b

# 4. Run fastevolve as usual
from fastevolve import Config, Controller
from fastevolve.llm_ensemble import ModelConfig

INITIAL = "def solve(x):\n    return x\n"

def correctness(p):
    ns = {}
    try: exec(p.code, ns)
    except Exception: return 0.0
    fn = ns.get("solve")
    cases = [(2, 4), (3, 9), (4, 16), (5, 25)]
    return sum(1 for x, y in cases if fn and fn(x) == y) / len(cases)

cfg = Config()
cfg.iterations = 20
cfg.ensemble.models = [
    ModelConfig(name="qwen2.5:0.5b", provider="ollama",
                temperature=0.7, weight=1.0, role="fast"),
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
ollama pull gemma3:e4b
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
    ModelConfig(name="gemma3:e4b", provider="ollama", temperature=0.6, weight=1.0, role="fast"),
    ModelConfig(name="gpt-4o-mini", provider="openai", temperature=0.6, weight=1.0, role="fast"),
    ModelConfig(name="claude-opus-4-7", provider="anthropic", temperature=0.7, weight=1.0,
                role="deep", options={"max_tokens": 4096}),
]
```

`provider` defaults to `"ollama"`, so existing configs keep working unchanged.
