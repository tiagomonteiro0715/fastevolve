import hashlib
import itertools
import re
from dataclasses import dataclass, field
from typing import Optional

from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

from .checkpoint import Checkpointer
from .config import Config
from .prompt_sampler import PromptSampler, TemplateLibrary
from .llm_ensemble import LLMEnsemble
from .evaluator import Evaluator
from .program_database import ProgramDatabase, Program
from .program_database import program as _program
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
        self.checkpoint = Checkpointer(config.checkpoint_path)
        records = self.checkpoint.load()
        self.database = ProgramDatabase(config.database)
        if records:
            for prog, res in records:
                self.database.add(prog, res)
            _program._ids = itertools.count(max(self.database.by_id, default=-1) + 1)
            self._start = max(0, len(records) - 1)
            log.info("resumed: [bold]%d[/] records replayed (population=%d)",
                     len(records), len(self.database.by_id))
        else:
            seed = self.database.seed(initial_program)
            seed_result = self.evaluator.execute(seed)
            self.database.add(seed, seed_result)
            self.checkpoint.append((seed, seed_result))
            self._start = 0
        self._eval_cache: dict[str, object] = {}

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
            task = prog.add_task("run", total=self.config.iterations, completed=self._start,
                                 stage="init", fit=0.0, best=0.0)
            for step in range(self._start, self.config.iterations):
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
                key = hashlib.sha256(child.code.encode()).hexdigest()
                if key in self._eval_cache:
                    result = self._eval_cache[key]
                else:
                    with span("evaluate"):
                        result = self.evaluator.execute(child)
                    self._eval_cache[key] = result
                self.database.add(child, result)
                self.ensemble.feedback(child.fitness > parent.fitness)
                if best is None or child.fitness > best.fitness:
                    best = child
                prog.update(task, advance=1, fit=child.fitness,
                            best=best.fitness if best else 0.0)
                self.checkpoint.append((child, result))
        self.checkpoint.close()
        t = timings()
        log.info("done. population=%d best=%.3f", len(self.database.by_id),
                 best.fitness if best else 0.0)
        for name, s in t.items():
            log.info("[cyan]%-9s[/] n=%-4d total=%6.2fs avg=%.3fs",
                     name, s["n"], s["total"], s["avg"])
        return RunResult(best=best, iterations=self.config.iterations,
                         stats={"population": len(self.database.by_id), "timings": t})
