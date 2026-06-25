# fastevolve

Minimal open-source AlphaEvolve: LLM-driven program evolution with MAP-Elites islands, cascade evaluation, and a local Ollama ensemble.

## Install

```bash
uv sync
```

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
