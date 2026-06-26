from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .config import Config
from .controller import Controller, RunResult, apply_diff
from .sandbox import run_sandboxed

try:
    __version__ = _pkg_version("fastevolve")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["Config", "Controller", "RunResult", "apply_diff", "run_sandboxed", "__version__"]
