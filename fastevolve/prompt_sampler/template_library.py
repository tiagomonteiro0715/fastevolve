from pathlib import Path

DEFAULT = """You are evolving a Python program. Improve the parent program.

# Parent
{parent}

# Inspirations
{inspirations}

Return ONLY the full replacement Python code (no prose, no fences)."""


class TemplateLibrary:
    def __init__(self):
        self.templates = {"default": DEFAULT}
        self.system_prompts: dict[str, str] = {}

    def load(self, path):
        for f in Path(path).glob("*.txt"):
            self.templates[f.stem] = f.read_text(encoding="utf-8")

    def get(self, name): return self.templates[name]

    def render(self, name, **vars): return self.templates[name].format(**vars)
