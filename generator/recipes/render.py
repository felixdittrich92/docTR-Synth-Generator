# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Crash-safe, resumable multi-process rendering of dataset parts.

A *part* is a folder ``<part>/images`` plus ``<part>/progress.jsonl``. Every
finished chunk of samples is appended to the progress file immediately, so an
interrupted run loses at most the chunks that were in flight: a restart reads
the progress file, skips what is done and renders the rest. Callers turn a
complete part into its final form (``labels.json``) and move it into place
atomically.
"""

import json
import os
import random
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

__all__ = ["Worker", "make_pool", "render_part", "Progress", "atomic_write_json"]

_WORKER: "Worker | None" = None


class Worker:
    """One renderer per process. Subclasses implement :meth:`render`."""

    def render(self, payload: Any, path: str) -> Any | None:
        """Render ``payload`` to ``path``; return the label record or ``None`` to drop it."""
        raise NotImplementedError


def _init(factory: Callable[..., Worker], args: tuple) -> None:
    global _WORKER
    import numpy as np

    # Independent RNG streams per process, whatever the start method.
    random.seed(os.urandom(16))
    np.random.seed(int.from_bytes(os.urandom(4), "little"))
    _WORKER = factory(*args)


def _work(items: list[tuple[str, Any, str]]) -> list[tuple[str, Any | None]]:
    assert _WORKER is not None
    out = []
    for name, payload, path in items:
        try:
            record = _WORKER.render(payload, path)
        except Exception as e:  # keep going; the sample is simply left out of the labels
            print(f"render error ({name}): {e!r}", flush=True)
            record = None
        if record is None and os.path.exists(path):
            os.remove(path)  # a stale image from an interrupted earlier attempt
        out.append((name, record))
    return out


def make_pool(workers: int, factory: Callable[..., Worker], args: tuple) -> ProcessPoolExecutor:
    """A process pool whose processes each hold one ``factory(*args)`` worker."""
    return ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(factory, args))


class Progress:
    """Throughput / ETA line for a run of ``total`` samples."""

    def __init__(self, total: int, unit: str = "img"):
        self.total, self.unit, self.done, self.t0, self._last = total, unit, 0, time.time(), 0.0

    def add(self, n: int, label: str = "") -> None:
        """Count ``n`` more samples and refresh the line at most twice a second."""
        self.done += n
        now = time.time()
        if now - self._last < 0.5 and self.done < self.total:
            return
        self._last = now
        rate = self.done / max(1e-6, now - self.t0)
        eta = (self.total - self.done) / max(rate, 1e-9) / 3600
        print(
            f"\r  {label} {self.done:,}/{self.total:,} | {rate:,.1f} {self.unit}/s | ETA {eta:.1f} h ",
            end="",
            flush=True,
        )


def _read_progress(path: Path) -> dict[str, Any]:
    done: dict[str, Any] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                break  # a torn last line from a crash: everything before it is valid
            done[rec["f"]] = rec["r"]
    return done


def render_part(
    pool: ProcessPoolExecutor,
    items: Iterable[tuple[str, Any]],
    part: Path,
    chunk: int,
    progress: Progress | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Render ``(file name, payload)`` items into ``part``; resumes from ``progress.jsonl``.

    Returns ``{file name: record}`` for every sample that rendered (failed samples
    are recorded as ``None`` in the progress file and not retried).
    """
    (part / "images").mkdir(parents=True, exist_ok=True)
    log = part / "progress.jsonl"
    done = _read_progress(log)
    todo = [(name, payload, str(part / "images" / name)) for name, payload in items if name not in done]
    if progress:
        progress.add(len(done), label)
    # Rewrite the log without a possibly torn tail before appending to it.
    with open(log, "w", encoding="utf-8") as fh:
        for name, rec in done.items():
            fh.write(json.dumps({"f": name, "r": rec}, ensure_ascii=False, separators=(",", ":")) + "\n")
    futures = [pool.submit(_work, todo[i : i + chunk]) for i in range(0, len(todo), chunk)]
    with open(log, "a", encoding="utf-8") as fh:
        for fut in as_completed(futures):
            results = fut.result()
            for name, rec in results:
                done[name] = rec
                fh.write(json.dumps({"f": name, "r": rec}, ensure_ascii=False, separators=(",", ":")) + "\n")
            fh.flush()
            if progress:
                progress.add(len(results), label)
    return {name: rec for name, rec in done.items() if rec is not None}


def atomic_write_json(path: Path, data: Any, **kwargs) -> None:
    """Write JSON so a reader never sees a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, **kwargs), encoding="utf-8")
    os.replace(tmp, path)
