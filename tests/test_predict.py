"""Tests for USL prediction — synthetic data, GPU-free."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analysis import predict


def test_usl_recovers_known_params():
    # Generate throughput from a known USL, fit, check the fit reproduces it.
    lam, alpha, beta = 50.0, 0.03, 0.002
    conc = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    thr = predict.usl(conc, lam, alpha, beta)
    agg = pd.DataFrame({"concurrency": conc, "throughput_tok_s": thr})
    fit = predict.fit_usl(agg)
    # Predictions match the true curve closely across the range.
    got = fit.predict(conc)
    assert got == pytest.approx(thr, rel=0.02)
    assert fit.c_min == 1 and fit.c_max == 32


def test_knee_matches_analytic_optimum():
    lam, alpha, beta = 50.0, 0.03, 0.002
    conc = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
    agg = pd.DataFrame({"concurrency": conc,
                        "throughput_tok_s": predict.usl(conc, lam, alpha, beta)})
    fit = predict.fit_usl(agg)
    analytic = np.sqrt((1 - alpha) / beta)
    assert fit.knee == pytest.approx(analytic, rel=0.05)
    # The fitted knee really is near the numerical throughput maximum.
    grid = np.linspace(1, 200, 4000)
    assert grid[int(np.argmax(fit.predict(grid)))] == pytest.approx(fit.knee, rel=0.1)


def test_knee_nan_when_no_coherency():
    # beta≈0: pure saturation, no interior maximum -> knee undefined, not a huge number.
    conc = np.array([1, 2, 4, 8, 16], dtype=float)
    thr = predict.usl(conc, 40.0, 0.1, 0.0)
    agg = pd.DataFrame({"concurrency": conc, "throughput_tok_s": thr})
    fit = predict.fit_usl(agg)
    assert not np.isfinite(fit.knee)


def _synth_raw(tmp_path, per_point=200, seed=0):
    """Raw per-request parquet whose aggregate throughput follows a known USL."""
    rng = np.random.default_rng(seed)
    lam, alpha, beta = 50.0, 0.03, 0.002
    for c in (1, 2, 4, 8, 16):
        target_tps = float(predict.usl(np.array([c]), lam, alpha, beta)[0])
        out_tokens = 64
        # Space arrivals so total_out / duration ~= target throughput.
        duration = per_point * out_tokens / target_tps
        rows = []
        for i in range(per_point):
            itl = (20.0 + rng.gamma(2.0, 1.0, size=out_tokens - 1)).tolist()
            ttft = 60.0 + rng.normal(0, 3)
            rows.append({
                "request_id": i, "concurrency": c, "prompt_len_target": 512,
                "prompt_tokens": 512, "output_tokens": out_tokens, "ttft_ms": ttft,
                "itl_ms": json.dumps(itl), "e2e_ms": ttft + float(np.sum(itl)),
                "t_start": i * duration / per_point, "error": None,
            })
        pd.DataFrame(rows).to_parquet(tmp_path / f"raw_c{c}.parquet", index=False)


def test_predict_table_regions_and_band(tmp_path):
    from analysis import decompose

    _synth_raw(tmp_path)
    df = decompose.load_raw(tmp_path)
    table = predict.predict_table(df, [4, 8, 40], n_boot=200, seed=1)

    reg = dict(zip(table["concurrency"], table["region"], strict=True))
    assert reg[4] == "INTERPOLATED"   # inside measured 1..16
    assert reg[8] == "INTERPOLATED"
    assert reg[40] == "EXTRAPOLATED"  # beyond 16

    # Band must bracket the point estimate (shared throughput definition).
    for _, row in table.iterrows():
        assert row["band_lo"] <= row["throughput_tok_s"] <= row["band_hi"]

    # The band is wider in the extrapolation region than deep in the interpolation region.
    w = {row["concurrency"]: row["band_hi"] - row["band_lo"] for _, row in table.iterrows()}
    rel_interp = w[8] / table.set_index("concurrency").loc[8, "throughput_tok_s"]
    rel_extrap = w[40] / table.set_index("concurrency").loc[40, "throughput_tok_s"]
    assert rel_extrap > rel_interp
