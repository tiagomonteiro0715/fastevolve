import multiprocessing as mp
from queue import Empty
from typing import Any


def _worker(code: str, fn_name: str, args: tuple, q) -> None:
    try:
        ns: dict = {}
        exec(code, ns)
        fn = ns.get(fn_name)
        if not callable(fn):
            q.put(("err", f"{fn_name!r} not defined or not callable"))
            return
        q.put(("ok", fn(*args)))
    except Exception as e:
        q.put(("err", repr(e)))


def run_sandboxed(code: str, fn_name: str, *args, timeout: float = 5.0) -> Any:
    """Execute `code` in a fresh process and call `fn_name(*args)`.

    Returns the function's value, or None on timeout / error / non-picklable result.
    Kills runaway processes after `timeout` seconds.
    """
    ctx = mp.get_context()
    q: Any = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, fn_name, args, q), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    try:
        status, val = q.get_nowait()
    except Empty:
        return None
    return val if status == "ok" else None
