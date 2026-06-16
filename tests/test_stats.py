"""Unit tests for the analysis core.

These run anywhere (no GPU): each test builds a synthetic series with a *known*
property and asserts the estimator recovers it. This is what lets us trust the
methodology before it ever touches a real serving run.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis import stats


@pytest.fixture
def rng() -> np.random.Generator:
    # Fixed seed: determinism is a project rule, tests included.
    return np.random.default_rng(20240615)


# --------------------------------------------------------------------------- #
# percentile_summary
# --------------------------------------------------------------------------- #
def test_percentile_summary_known_values():
    s = stats.percentile_summary(list(range(1, 101)))  # 1..100
    assert s.n == 100
    assert s.min == 1.0 and s.max == 100.0
    assert s.p50 == pytest.approx(50.5, abs=0.5)
    assert s.p99 == pytest.approx(99.01, abs=0.5)
    assert s.p95 < s.p99 <= s.max


def test_percentile_summary_rejects_empty():
    with pytest.raises(ValueError):
        stats.percentile_summary([np.nan, np.inf])


def test_tail_ratio_heavy_tail():
    # Median ~10, top 2% are large spikes -> p99 lands in the tail, p99/p50 >> 1.
    base = np.full(1000, 10.0)
    base[:20] = 200.0  # 2% > the 1% that p99 measures, so p99 ~ 200
    assert stats.tail_ratio(base) > 5.0


# --------------------------------------------------------------------------- #
# detect_warmup (MSER-5)
# --------------------------------------------------------------------------- #
def test_warmup_truncates_decaying_transient(rng):
    # Big exponential transient (first ~100) settling onto a stationary level.
    n = 600
    transient = 80.0 * np.exp(-np.arange(n) / 25.0)
    steady = 20.0 + rng.normal(0, 1.0, n)
    series = transient + steady
    w = stats.detect_warmup(series)
    # Should discard a chunk inside the transient region, not zero, not everything.
    assert 20 <= w.cutoff_index <= 200
    assert 0.0 < w.fraction_discarded < 0.5


def test_warmup_keeps_stationary_series(rng):
    # No transient -> MSER should truncate little (near the front).
    series = 20.0 + rng.normal(0, 1.0, 600)
    w = stats.detect_warmup(series)
    assert w.fraction_discarded < 0.2


def test_warmup_short_series_keeps_all():
    w = stats.detect_warmup([1.0, 2.0, 3.0])
    assert w.cutoff_index == 0


# --------------------------------------------------------------------------- #
# autocorrelation
# --------------------------------------------------------------------------- #
def test_acf_recovers_ar1_coefficient(rng):
    phi = 0.6
    n = 20000
    x = np.zeros(n)
    noise = rng.normal(0, 1.0, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    r = stats.autocorrelation(x, max_lag=5)
    assert r.lag1 == pytest.approx(phi, abs=0.05)
    # AR(1) ACF decays geometrically: rho_2 ~ phi^2.
    assert r.acf[2] == pytest.approx(phi**2, abs=0.06)
    # Positive correlation inflates the effective variance.
    assert r.integrated_autocorr_time > 1.5


def test_acf_white_noise_within_band(rng):
    x = rng.normal(0, 1.0, 5000)
    r = stats.autocorrelation(x, max_lag=10)
    # Lag-1 of white noise should sit inside the significance band.
    assert abs(r.lag1) < r.conf95 * 3
    assert r.integrated_autocorr_time == pytest.approx(1.0, abs=0.5)


# --------------------------------------------------------------------------- #
# convergence_window (Allan variance)
# --------------------------------------------------------------------------- #
def test_allan_dev_decreases_for_white_noise(rng):
    # White noise: Allan deviation falls ~ tau^-1/2, so longer averaging helps.
    x = 5.0 + rng.normal(0, 1.0, 4000)
    c = stats.convergence_window(x)
    assert len(c.taus) >= 5
    # Deviation at the smallest window should exceed that at the largest.
    assert c.allan_dev[0] > c.allan_dev[-1]
    assert c.convergence_window >= c.taus[len(c.taus) // 2]


def test_allan_dev_slope_near_minus_half(rng):
    x = rng.normal(0, 1.0, 8000)
    c = stats.convergence_window(x)
    taus = np.asarray(c.taus, dtype=float)
    dev = np.asarray(c.allan_dev, dtype=float)
    # Fit log(dev) vs log(tau) over the white-noise region; slope ~ -0.5.
    mask = taus <= taus.max() / 4
    slope = np.polyfit(np.log(taus[mask]), np.log(dev[mask]), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.15)


# --------------------------------------------------------------------------- #
# characterize (end to end)
# --------------------------------------------------------------------------- #
def test_characterize_pipeline(rng):
    n = 800
    transient = 60.0 * np.exp(-np.arange(n) / 20.0)
    series = 25.0 + transient + rng.normal(0, 2.0, n)
    out = stats.characterize(series)
    assert out["summary"]["p99"] >= out["summary"]["p50"]
    assert out["warmup"]["cutoff_index"] >= 0
    assert out["convergence"] is not None
    assert "integrated_autocorr_time" in out["autocorrelation"]
