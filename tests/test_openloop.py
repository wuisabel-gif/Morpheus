"""Tests for the open-loop (Poisson) mode, GPU-free.

The arrival schedule is a pure function (tested directly); the analysis path is
exercised on synthetic raw_r*.parquet the same way test_pipeline.py exercises the
closed-loop path — synthetic numbers never touch the committed results/ tree.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analysis import decompose
from harness import sweep


# --------------------------------------------------------------------------- #
# Pure arrival schedule
# --------------------------------------------------------------------------- #
def test_poisson_offsets_deterministic_and_monotone():
    a = sweep.poisson_offsets(200, rate_rps=2.0, seed=7)
    b = sweep.poisson_offsets(200, rate_rps=2.0, seed=7)
    assert a == b  # a rerun replays the identical arrival schedule
    assert all(x < y for x, y in zip(a, a[1:], strict=False))  # strictly increasing


def test_poisson_offsets_mean_gap_matches_rate():
    rate = 4.0
    offsets = np.asarray(sweep.poisson_offsets(20_000, rate_rps=rate, seed=0))
    gaps = np.diff(np.concatenate(([0.0], offsets)))
    # Exponential(1/rate) gaps: mean within a few percent at this n.
    assert np.mean(gaps) == pytest.approx(1.0 / rate, rel=0.05)


def test_poisson_offsets_distinct_across_rates_and_seeds():
    assert sweep.poisson_offsets(50, 1.0, seed=0) != sweep.poisson_offsets(50, 2.0, seed=0)
    assert sweep.poisson_offsets(50, 1.0, seed=0) != sweep.poisson_offsets(50, 1.0, seed=1)


def test_poisson_offsets_rejects_bad_args():
    with pytest.raises(ValueError):
        sweep.poisson_offsets(0, 1.0)
    with pytest.raises(ValueError):
        sweep.poisson_offsets(10, 0.0)


# --------------------------------------------------------------------------- #
# Synthetic open-loop raw data through the real analysis path
# --------------------------------------------------------------------------- #
def _synth_open_loop_raw(tmp_path, rates=(0.5, 1.0, 2.0, 4.0)):
    rng = np.random.default_rng(3)
    n = 80
    for rate in rates:
        base_itl = 20.0 + 3.0 * rate  # queueing pushes decode latency up with load
        tail_scale = 1.0 + (rate / 2.0) ** 2
        rows = []
        t = 0.0
        for i in range(n):
            warmup = 30.0 * np.exp(-i / 6.0)
            out_tokens = 64
            itl = (
                base_itl + warmup + rng.gamma(2.0, tail_scale, size=out_tokens - 1)
            ).tolist()
            ttft = 60.0 + 10.0 * rate + warmup + rng.normal(0, 3)
            e2e_ms = ttft + float(np.sum(itl))
            rows.append(
                {
                    "request_id": i,
                    "concurrency": 0,  # emergent, not pinned, in open loop
                    "prompt_len_target": 512,
                    "prompt_tokens": 512,
                    "output_tokens": out_tokens,
                    "ttft_ms": ttft,
                    "itl_ms": json.dumps(itl),
                    "e2e_ms": e2e_ms,
                    "t_start": t,
                    "error": None,
                    "arrival_rate_rps": rate,
                }
            )
            t += 1.0 / rate  # arrivals paced by offered load, not completions
        pd.DataFrame(rows).to_parquet(tmp_path / f"raw_r{rate:g}.parquet", index=False)


def test_rate_aggregate_and_knee(tmp_path):
    _synth_open_loop_raw(tmp_path)
    df = decompose.load_raw(tmp_path, pattern="raw_r*.parquet")
    agg = decompose.aggregate_rate_sweep(df)

    assert list(agg["arrival_rate_rps"]) == sorted(agg["arrival_rate_rps"])
    # Tail widens with offered load, same signature as the closed loop.
    ratio_lo = agg.iloc[0]["itl_p99"] / agg.iloc[0]["itl_p50"]
    ratio_hi = agg.iloc[-1]["itl_p99"] / agg.iloc[-1]["itl_p50"]
    assert ratio_hi > ratio_lo
    # Error rate column exists and is clean on this synthetic data.
    assert (agg["error_rate"] == 0).all()

    knee = decompose.find_knee(agg, x_col="arrival_rate_rps")
    assert knee in set(agg["arrival_rate_rps"])


def test_closed_loop_glob_ignores_open_loop_files(tmp_path):
    # A directory holding only open-loop data must not satisfy the closed-loop load.
    _synth_open_loop_raw(tmp_path, rates=(1.0,))
    with pytest.raises(FileNotFoundError):
        decompose.load_raw(tmp_path)  # default pattern raw_c*
    df = decompose.load_raw(tmp_path, pattern="raw_r*.parquet")
    assert len(df) == 80
