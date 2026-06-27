import json
import subprocess
import sys
from typing import Any


def run_sandboxed(code: str, fn_name: str, *args, timeout: float = 5.0) -> Any:
    """Execute `code` in a fresh `python -c` subprocess and call `fn_name(*args)`.

    Returns the function's value (must be JSON-serializable), or None on timeout,
    crash, missing function, or unserializable result. Kills runaway processes
    after `timeout` seconds.

    Args must be JSON-serializable (int, float, str, list, dict, bool, None).
    """
    runner = (
        f"import json,sys;ns={{}};exec({code!r},ns);"
        f"fn=ns.get({fn_name!r});"
        f"sys.exit(2) if not callable(fn) else print(json.dumps(fn(*{list(args)!r})))"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True, timeout=timeout, text=True,
        )
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return None
