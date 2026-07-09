"""Derive aggregates from raw per-request data.

Everything here is downstream of ``results/raw/`` and never hand-edited. The two
phases are kept apart: TTFT is prefill (compute-bound), the ITL stream is decode
(memory-bound). Warmup is removed via the MSER-5 detector before any aggregate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import stats


def load_raw(raw_dir: Path | str, pattern: str = "raw_c*.parquet") -> pd.DataFrame:
    """Load and concatenate raw parquet files matching ``pattern``, parsing ITL lists.

    Closed-loop points are ``raw_c*.parquet``; open-loop (Poisson) points are
    ``raw_r*.parquet`` — pass ``pattern="raw_r*.parquet"`` for those.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {raw_dir}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["itl_ms"] = df["itl_ms"].apply(lambda s: json.loads(s) if isinstance(s, str) else list(s))
    return df


def _ordered_steady(group: pd.DataFrame) -> pd.DataFrame:
    """Sort a concurrency group by start time and drop MSER-detected warmup."""
    g = group.sort_values("t_start")
    # Use per-request mean ITL as the steady-state signal (decode is the phase under load).
    signal = g["itl_ms"].apply(lambda x: float(np.mean(x)) if len(x) else np.nan).to_numpy()
    finite = signal[np.isfinite(signal)]
    if finite.size >= 10:
        cutoff = stats.detect_warmup(finite).cutoff_index
        return g.iloc[cutoff:]
    return g


def _group_metrics(group: pd.DataFrame) -> dict[str, object]:
    """Per-sweep-point metrics (warmup removed) shared by both load modes."""
    full = group.sort_values("t_start")
    steady = _ordered_steady(group)
    discarded = len(full) - len(steady)

    # Throughput over the steady-state wall-clock for this point.
    ends = steady["t_start"] + steady["e2e_ms"] / 1e3
    duration = float(ends.max() - steady["t_start"].min()) if len(steady) else float("nan")
    total_out = float(steady["output_tokens"].sum())
    throughput = total_out / duration if duration and duration > 0 else float("nan")

    # Errored requests get nan TTFT in the raw rows; their share is itself a
    # saturation signal, so it is reported rather than silently dropped.
    n_errors = int(steady["error"].notna().sum()) if "error" in steady else 0

    ttft = steady["ttft_ms"].to_numpy()
    ttft = ttft[np.isfinite(ttft)]
    itl_pool = np.array([v for lst in steady["itl_ms"] for v in lst], dtype=float)

    ttft_sum = stats.percentile_summary(ttft) if ttft.size else None
    itl_sum = stats.percentile_summary(itl_pool) if itl_pool.size else None
    lag1 = stats.autocorrelation(itl_pool).lag1 if itl_pool.size > 2 else float("nan")

    return {
        "n_requests": int(len(steady)),
        "error_rate": n_errors / len(steady) if len(steady) else float("nan"),
        "throughput_tok_s": throughput,
        "ttft_p50": ttft_sum.p50 if ttft_sum else float("nan"),
        "ttft_p99": ttft_sum.p99 if ttft_sum else float("nan"),
        "itl_p50": itl_sum.p50 if itl_sum else float("nan"),
        "itl_p95": itl_sum.p95 if itl_sum else float("nan"),
        "itl_p99": itl_sum.p99 if itl_sum else float("nan"),
        "itl_lag1_acf": lag1,
        "warmup_discarded": discarded,
    }


def aggregate_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """One row per concurrency: throughput + TTFT/ITL distributions (warmup removed)."""
    rows = [
        {"concurrency": int(c), **_group_metrics(group)}
        for c, group in df.groupby("concurrency")
    ]
    return pd.DataFrame(rows).sort_values("concurrency").reset_index(drop=True)


def aggregate_rate_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Open-loop counterpart of :func:`aggregate_sweep`: one row per Poisson rate.

    Groups by ``arrival_rate_rps`` (rows written by ``run_open_loop_sweep``). The
    same warmup removal and distribution summaries apply; only the independent
    variable changes from pinned concurrency to offered load.
    """
    if "arrival_rate_rps" not in df.columns:
        raise ValueError("no arrival_rate_rps column — is this open-loop data (raw_r*)?")
    rows = [
        {"arrival_rate_rps": float(r), **_group_metrics(group)}
        for r, group in df.groupby("arrival_rate_rps")
    ]
    return pd.DataFrame(rows).sort_values("arrival_rate_rps").reset_index(drop=True)


@dataclass(frozen=True)
class PrefillDecodeSplit:
    prefill_ttft: dict[str, float | int]
    decode_itl: dict[str, float | int]
    decode_tokens_per_s: float  # 1000 / median ITL

    def as_dict(self) -> dict[str, object]:
        return {
            "prefill_ttft": self.prefill_ttft,
            "decode_itl": self.decode_itl,
            "decode_tokens_per_s": self.decode_tokens_per_s,
        }


def prefill_decode_split(df: pd.DataFrame, concurrency: int = 1) -> PrefillDecodeSplit:
    """Single-stream prefill vs decode summary at the given concurrency (default 1).

    The two-machines claim: TTFT (one prefill of the whole prompt) vs ITL (the
    per-token decode stream). Reported as distributions, not means.
    """
    g = df[df["concurrency"] == concurrency]
    if g.empty:
        raise ValueError(f"no rows at concurrency={concurrency}")
    g = _ordered_steady(g)
    ttft = g["ttft_ms"].to_numpy()
    ttft = ttft[np.isfinite(ttft)]
    itl = np.array([v for lst in g["itl_ms"] for v in lst], dtype=float)
    itl_sum = stats.percentile_summary(itl)
    return PrefillDecodeSplit(
        prefill_ttft=stats.percentile_summary(ttft).as_dict(),
        decode_itl=itl_sum.as_dict(),
        decode_tokens_per_s=1000.0 / itl_sum.p50 if itl_sum.p50 else float("nan"),
    )


def find_knee(agg: pd.DataFrame, x_col: str = "concurrency") -> float:
    """Knee of the throughput curve (Kneedle, concave-increasing).

    Normalize both axes to [0,1]; for a diminishing-returns curve the knee is the
    point furthest above the chord from first to last (max of y_norm - x_norm). That
    x (concurrency, or arrival rate for open-loop data) is the honest operating
    point — past it, throughput stalls while tail latency climbs.
    """
    a = agg.sort_values(x_col)
    x = a[x_col].to_numpy(dtype=float)
    y = a["throughput_tok_s"].to_numpy(dtype=float)
    if len(x) < 3:
        return float(x[-1])
    xn = (x - x.min()) / (np.ptp(x) or 1.0)
    yn = (y - y.min()) / (np.ptp(y) or 1.0)
    return float(x[int(np.argmax(yn - xn))])
