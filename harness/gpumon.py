"""Background GPU telemetry sampler.

The README claims decode is memory-bandwidth-bound. That claim must be *measured*,
not asserted: this module samples ``nvidia-smi`` on a background thread while the
sweep runs. Every sample is timestamped with the same ``time.perf_counter`` clock
as each request's ``t_start``, so utilization can be aligned with sweep points and
request phases after the fact.

Two fields carry the roofline story:

* ``utilization.gpu`` — % of the sample window in which at least one SM was busy.
  High during prefill (compute pressure).
* ``utilization.memory`` — % of the window in which the memory controller was busy.
  This is the bandwidth-bound signal for decode.

Samples land in ``results/raw/gpu_util.parquet`` (one row per GPU per tick, tagged
with the active sweep-point label) and a per-label summary goes into
``run_meta.json``. Parsing is a pure function and the sampler takes an injectable
query callable, so the whole module is unit-testable without hardware.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

_QUERY_FIELDS = "utilization.gpu,utilization.memory,memory.used,power.draw"

# One reading = (util_gpu_pct, util_mem_pct, memory_used_mib, power_w) for one GPU.
Reading = tuple[float, float, float, float]
QueryFn = Callable[[], list[Reading]]


@dataclass(frozen=True)
class GpuSample:
    t: float  # time.perf_counter() at sample — same clock as RequestRecord.t_start
    gpu_index: int
    util_gpu_pct: float  # SM-busy fraction: compute pressure (prefill)
    util_mem_pct: float  # memory-controller-busy fraction: bandwidth pressure (decode)
    memory_used_mib: float
    power_w: float
    label: str  # active sweep-point tag, e.g. "c8" or "r2"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _field(raw: str) -> float:
    """One CSV field -> float; '[N/A]' and friends become nan, not a lost sample."""
    try:
        return float(raw.strip())
    except ValueError:
        return float("nan")


def parse_smi_csv(output: str) -> list[Reading]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` output.

    Returns one reading per GPU line. Lines with the wrong field count are dropped;
    individual unparseable fields (e.g. ``[N/A]`` power on some boards) become nan so
    the rest of the sample survives. Pure function — this is the tested surface.
    """
    readings: list[Reading] = []
    for line in output.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 4:
            continue
        util_gpu, util_mem, mem_used, power = (_field(p) for p in parts)
        readings.append((util_gpu, util_mem, mem_used, power))
    return readings


def smi_query() -> list[Reading]:
    """One-shot ``nvidia-smi`` telemetry query; empty list if unavailable or failing."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    try:
        out = subprocess.run(
            [smi, f"--query-gpu={_QUERY_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    return parse_smi_csv(out.stdout)


class GpuSampler:
    """Sample GPU telemetry on a daemon thread for the lifetime of a ``with`` block.

    Usage::

        sampler = GpuSampler(interval_s=0.25)
        with sampler:
            for c in concurrencies:
                sampler.set_label(f"c{c}")
                ...run the sweep point...
        sampler.to_parquet(out_dir / "gpu_util.parquet")

    A custom ``query_fn`` can be injected for tests. If the query yields nothing
    (no nvidia-smi), the sampler is inert: no samples, no file — never fabricated
    utilization numbers.
    """

    def __init__(self, interval_s: float = 0.25, query_fn: QueryFn | None = None) -> None:
        self.interval_s = interval_s
        self._query: QueryFn = query_fn if query_fn is not None else smi_query
        self._samples: list[GpuSample] = []
        self._label = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    def set_label(self, label: str) -> None:
        """Tag subsequent samples with the active sweep point (e.g. ``c8``)."""
        with self._lock:
            self._label = label

    @property
    def samples(self) -> list[GpuSample]:
        with self._lock:
            return list(self._samples)

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        readings = self._query()
        now = time.perf_counter()
        if not readings:
            return
        with self._lock:
            label = self._label
            self._samples.extend(
                GpuSample(
                    t=now,
                    gpu_index=i,
                    util_gpu_pct=r[0],
                    util_mem_pct=r[1],
                    memory_used_mib=r[2],
                    power_w=r[3],
                    label=label,
                )
                for i, r in enumerate(readings)
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuSampler:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gpumon", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, 4 * self.interval_s))
            self._thread = None

    # ------------------------------------------------------------------ #
    def to_parquet(self, path: Path | str) -> Path | None:
        """Write all samples to parquet; returns None (writes nothing) if empty."""
        rows = self.samples
        if not rows:
            return None
        import pandas as pd

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([s.as_dict() for s in rows]).to_parquet(path, index=False)
        return path

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Per-label mean utilization — the run_meta-sized digest of the timeseries."""
        import numpy as np

        by_label: dict[str, list[GpuSample]] = {}
        for s in self.samples:
            by_label.setdefault(s.label, []).append(s)
        out: dict[str, dict[str, float | int]] = {}
        for label, group in by_label.items():
            gpu = np.array([s.util_gpu_pct for s in group], dtype=float)
            mem = np.array([s.util_mem_pct for s in group], dtype=float)
            out[label] = {
                "n_samples": len(group),
                "util_gpu_pct_mean": float(np.nanmean(gpu)) if gpu.size else float("nan"),
                "util_mem_pct_mean": float(np.nanmean(mem)) if mem.size else float("nan"),
            }
        return out
