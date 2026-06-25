from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .config import Config
from .controller import Controller, RunResult, apply_diff

try:
    __version__ = _pkg_version("fastevolve")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["Config", "Controller", "RunResult", "apply_diff", "__version__"]
