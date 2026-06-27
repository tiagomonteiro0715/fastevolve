"""Demo: evolve a Python `solve(x)` function with gemma4:e4b on Ollama.

Run from the project root (where the `fastevolve/` package sits):
    python main.py
    # or:  uv run python main.py
"""
from fastevolve import Config, Controller
from fastevolve.llm_ensemble import ModelConfig

MODEL = "gemma4:e2b"

INITIAL_PROGRAM = '''def solve(x):
    total = 0
    for i in range(x):
        total += x
    if x < 0:
        total = -total
    return total + 1
'''


def syntax_score(program) -> float:
    try:
        compile(program.code, "<prog>", "exec")
        return 1.0
    except SyntaxError:
        return 0.0


def correctness_score(program) -> float:
    ns: dict = {}
    try:
        exec(program.code, ns)
    except Exception:
        return 0.0
    fn = ns.get("solve")
    if not callable(fn):
        return 0.0
    cases = [(2, 4), (3, 9), (4, 16), (5, 25)]
    hits = sum(1 for x, y in cases if _safe(fn, x) == y)
    return hits / len(cases)


def _safe(fn, x):
    try:
        return fn(x)
    except Exception:
        return None


def main():
    cfg = Config()
    cfg.iterations = 20
    cfg.ensemble.models = [
        ModelConfig(name=MODEL, temperature=0.6, weight=1.0, role="fast",
                    num_ctx=8192, flash_attention=True),
        ModelConfig(name=MODEL, temperature=1.0, weight=1.0, role="deep",
                    num_ctx=8192, flash_attention=True),
    ]
    cfg.evaluator.cascade = [(syntax_score, 1.0), (correctness_score, 0.0)]
    cfg.prompt.system_prompts = {MODEL: "You are an expert Python programmer. Output only Python code."}

    result = Controller(cfg, initial_program=INITIAL_PROGRAM).run()
    print("\n=== Best program ===")
    print(result.best.code if result.best else "(none)")
    print("Stats:", result.stats)


if __name__ == "__main__":
    main()
