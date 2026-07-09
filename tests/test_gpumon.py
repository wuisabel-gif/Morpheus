"""Tests for the GPU telemetry sampler — no hardware required.

``parse_smi_csv`` is a pure function; the sampler takes an injected query callable,
so the thread lifecycle, labeling, parquet output, and per-label summary are all
exercised with synthetic readings.
"""

from __future__ import annotations

import math
import time

import pandas as pd

from harness import gpumon


# --------------------------------------------------------------------------- #
# Pure parsing
# --------------------------------------------------------------------------- #
def test_parse_smi_csv_single_gpu():
    out = "87, 63, 21504, 285.17\n"
    readings = gpumon.parse_smi_csv(out)
    assert readings == [(87.0, 63.0, 21504.0, 285.17)]


def test_parse_smi_csv_multi_gpu_and_na_field():
    # Second GPU reports [N/A] power (seen on some boards): field -> nan, sample kept.
    out = "12, 95, 40000, 300\n34, 88, 40960, [N/A]\n"
    readings = gpumon.parse_smi_csv(out)
    assert len(readings) == 2
    assert readings[0] == (12.0, 95.0, 40000.0, 300.0)
    assert readings[1][:3] == (34.0, 88.0, 40960.0)
    assert math.isnan(readings[1][3])


def test_parse_smi_csv_drops_malformed_lines():
    out = "not a reading\n50, 60, 1000, 100\n1, 2, 3\n"
    assert gpumon.parse_smi_csv(out) == [(50.0, 60.0, 1000.0, 100.0)]


def test_parse_smi_csv_empty():
    assert gpumon.parse_smi_csv("") == []


# --------------------------------------------------------------------------- #
# Sampler with an injected query (no nvidia-smi)
# --------------------------------------------------------------------------- #
def _wait_for(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise TimeoutError("sampler did not produce expected samples in time")


def test_sampler_collects_labels_and_summary(tmp_path):
    fake = lambda: [(80.0, 20.0, 10000.0, 250.0)]  # noqa: E731 - tiny test stub
    sampler = gpumon.GpuSampler(interval_s=0.01, query_fn=fake)
    with sampler:
        sampler.set_label("c1")
        _wait_for(lambda: any(s.label == "c1" for s in sampler.samples))
        sampler.set_label("c8")
        _wait_for(lambda: any(s.label == "c8" for s in sampler.samples))

    labels = {s.label for s in sampler.samples}
    assert {"c1", "c8"} <= labels

    summary = sampler.summary()
    assert summary["c1"]["util_gpu_pct_mean"] == 80.0
    assert summary["c8"]["util_mem_pct_mean"] == 20.0
    assert summary["c1"]["n_samples"] >= 1

    # Samples share the perf_counter clock with RequestRecord.t_start.
    now = time.perf_counter()
    assert all(0 < s.t <= now for s in sampler.samples)

    path = sampler.to_parquet(tmp_path / "gpu_util.parquet")
    assert path is not None and path.exists()
    df = pd.read_parquet(path)
    assert set(df.columns) >= {"t", "gpu_index", "util_gpu_pct", "util_mem_pct", "label"}
    assert len(df) == len(sampler.samples)


def test_sampler_inert_without_readings(tmp_path):
    # No GPU -> query yields nothing -> no samples, no file, no fabricated numbers.
    sampler = gpumon.GpuSampler(interval_s=0.01, query_fn=list)
    with sampler:
        time.sleep(0.05)
    assert sampler.samples == []
    assert sampler.summary() == {}
    assert sampler.to_parquet(tmp_path / "gpu_util.parquet") is None
    assert not (tmp_path / "gpu_util.parquet").exists()
