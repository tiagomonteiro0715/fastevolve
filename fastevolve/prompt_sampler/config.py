from dataclasses import dataclass, field


@dataclass(kw_only=True)
class PromptConfig:
    template: str = "default"
    num_inspirations: int = 3
    temperature: float = 0.7
    system_prompts: dict = field(default_factory=dict)
