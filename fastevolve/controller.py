import re
from dataclasses import dataclass, field
from typing import Optional

from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

from .config import Config
from .prompt_sampler import PromptSampler, TemplateLibrary
from .llm_ensemble import LLMEnsemble
from .evaluator import Evaluator
from .program_database import ProgramDatabase, Program
from .telemetry import setup, span, timings, log


@dataclass(kw_only=True)
class RunResult:
    best: Optional[Program]
    iterations: int
    stats: dict = field(default_factory=dict)


def apply_diff(parent_program: Program, diff: str) -> Program:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", diff, re.DOTALL)
    code = (m.group(1) if m else diff).strip()
    if not code:
        code = parent_program.code
    return Program(code=code, parent_id=parent_program.id, island=parent_program.island,
                   generation=parent_program.generation + 1)


class Controller:
    def __init__(self, config: Config, *, initial_program: str):
        setup()
        self.config = config
        lib = TemplateLibrary()
        self.sampler = PromptSampler(config.prompt, library=lib)
        self.ensemble = LLMEnsemble(config.ensemble, system_prompts=lib.system_prompts)
        self.evaluator = Evaluator(config.evaluator)
        self.database = ProgramDatabase(config.database)
        seed = self.database.seed(initial_program)
        self.database.add(seed, self.evaluator.execute(seed))

    def run(self) -> RunResult:
        best: Optional[Program] = None
        cols = [
            TextColumn("[bold cyan]evolve[/]"),
            BarColumn(), MofNCompleteColumn(),
            TextColumn("[yellow]{task.fields[stage]:>8}[/]"),
            TextColumn("fit=[green]{task.fields[fit]:.3f}[/]"),
            TextColumn("best=[magenta]{task.fields[best]:.3f}[/]"),
            TimeElapsedColumn(), TimeRemainingColumn(),
        ]
        with Progress(*cols, transient=False) as prog:
            task = prog.add_task("run", total=self.config.iterations, stage="init", fit=0.0, best=0.0)
            for _ in range(self.config.iterations):
                prog.update(task, stage="sample")
                with span("sample"):
                    parent, insp = self.database.sample()
                prog.update(task, stage="prompt")
                with span("prompt"):
                    prompt = self.sampler.build(parent, insp)
                prog.update(task, stage="generate")
                with span("generate"):
                    diff = self.ensemble.generate(prompt)
                with span("apply"):
                    child = apply_diff(parent, diff)
                    child.island = parent.island
                prog.update(task, stage="evaluate")
                with span("evaluate"):
                    result = self.evaluator.execute(child)
                self.database.add(child, result)
                if best is None or child.fitness > best.fitness:
                    best = child
                prog.update(task, advance=1, fit=child.fitness,
                            best=best.fitness if best else 0.0)
        t = timings()
        log.info("done. population=%d best=%.3f", len(self.database.by_id),
                 best.fitness if best else 0.0)
        for name, s in t.items():
            log.info("[cyan]%-9s[/] n=%-4d total=%6.2fs avg=%.3fs",
                     name, s["n"], s["total"], s["avg"])
        return RunResult(best=best, iterations=self.config.iterations,
                         stats={"population": len(self.database.by_id), "timings": t})
