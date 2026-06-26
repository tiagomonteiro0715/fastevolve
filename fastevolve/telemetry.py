import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager

from rich.logging import RichHandler

log = logging.getLogger("fastevolve")
_TIMINGS: dict[str, list[float]] = defaultdict(list)


def setup() -> None:
    if logging.getLogger().handlers:
        return
    level = logging.WARNING if os.getenv("FASTEVOLVE_QUIET") else logging.INFO
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True,
                              show_path=False, markup=True)],
    )


@contextmanager
def span(name: str):
    t = time.perf_counter()
    try:
        yield
    except Exception:
        log.exception("[red]%s failed after %.2fs[/]", name, time.perf_counter() - t)
        raise
    finally:
        _TIMINGS[name].append(time.perf_counter() - t)


def timings() -> dict[str, dict[str, float]]:
    return {k: {"n": len(v), "total": sum(v), "avg": sum(v) / len(v)} for k, v in _TIMINGS.items()}
