"""Reusable helpers: hourly load parquet I/O, forecast metrics, and matplotlib plots."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

DEFAULT_TZ = "Europe/Berlin"
DEFAULT_HOURLY_PARQUET_REL = Path("data/raw/de_lu_load_hourly.parquet")
DEFAULT_VALUE_COL = "load_mw"


def resolve_hourly_parquet_path(
    parquet_path: str | Path | None = None,
    *,
    extra_candidates: tuple[str | Path, ...] = (),
) -> Path:
    """Return the first existing path among explicit, env, default, and extra candidates."""
    candidates: list[Path] = []
    if parquet_path is not None:
        candidates.append(Path(parquet_path))
    env = os.environ.get("ENERGY_TS_LOAD_PARQUET")
    if env:
        candidates.append(Path(env.strip()))
    candidates.append(DEFAULT_HOURLY_PARQUET_REL)
    candidates.extend(Path(p) for p in extra_candidates)

    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.is_absolute() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find hourly load parquet. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def load_hourly_load_series(
    parquet_path: str | Path | None = None,
    *,
    tz: str = DEFAULT_TZ,
    value_col: str = DEFAULT_VALUE_COL,
    extra_path_candidates: tuple[str | Path, ...] = (),
) -> pd.Series:
    """Load `value_col` from parquet: sort, hourly frequency, interpolate gaps, float dtype."""
    path = resolve_hourly_parquet_path(parquet_path, extra_candidates=extra_path_candidates)
    load_hourly = pd.read_parquet(path)
    idx = load_hourly.index
    if not isinstance(idx, pd.DatetimeIndex):
        load_hourly.index = pd.to_datetime(idx, utc=True)
    if load_hourly.index.tz is None:
        load_hourly.index = load_hourly.index.tz_localize(
            tz, ambiguous="infer", nonexistent="shift_forward"
        )
    else:
        load_hourly.index = load_hourly.index.tz_convert(tz)

    return (
        load_hourly[value_col]
        .sort_index()
        .asfreq("1h")
        .interpolate(limit_direction="both")
        .astype(float)
    )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = np.finfo(float).eps
    return float(
        np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0
    )


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > np.finfo(float).eps
    return float(np.mean(200.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "sMAPE_%": smape(y_true, y_pred),
    }


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def pinball(y_true: np.ndarray, y_q: np.ndarray, q: float) -> float:
    """Pinball / quantile loss at level q ∈ (0, 1). Lower is better."""
    diff = y_true - y_q
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def rolling_origin_eval(
    model_fn: Callable[[pd.Series, int], np.ndarray],
    series: pd.Series,
    h: int,
    n_splits: int,
    step: int,
    window: str = "sliding",
    window_size: int | None = None,
    min_train_size: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.Series]]:
    """Rolling-origin backtest: refit `model_fn` on each fold and forecast `h` ahead.

    Parameters
    ----------
    model_fn :
        ``(train: pd.Series, h: int) -> np.ndarray`` of length ``h``; full refit each call.
    series :
        Full series with a monotonic DatetimeIndex (regular freq recommended).
    h :
        Forecast horizon in observations.
    n_splits :
        Number of folds; non-overlapping when ``step >= h``.
    step :
        Spacing between consecutive fold cuts.
    window :
        ``\"expanding\"`` or ``\"sliding\"``.
    window_size :
        Required when ``window=\"sliding\"``.
    min_train_size :
        Minimum train length for expanding window (default ``5 * h``).

    Returns
    -------
    per_fold, aggregate, forecasts
    """
    if window not in {"expanding", "sliding"}:
        raise ValueError("window must be 'expanding' or 'sliding'.")
    if window == "sliding" and window_size is None:
        raise ValueError("window_size is required for sliding window.")
    if min_train_size is None:
        min_train_size = 5 * h

    n = len(series)
    rows: list[dict] = []
    forecasts: dict[int, pd.Series] = {}

    for k in range(n_splits):
        test_end = n - k * step
        test_start = test_end - h
        train_end = test_start
        if train_end <= 0:
            raise ValueError(f"Fold {k}: train_end <= 0 — not enough history.")

        if window == "expanding":
            train_start = 0
            if train_end < min_train_size:
                raise ValueError(
                    f"Fold {k}: train length {train_end} below min_train_size {min_train_size}."
                )
        else:
            train_start = train_end - int(window_size)
            if train_start < 0:
                raise ValueError(
                    f"Fold {k}: not enough history for sliding window_size={window_size}."
                )

        train = series.iloc[train_start:train_end]
        test = series.iloc[test_start:test_end]

        y_pred = np.asarray(model_fn(train, h), dtype=float)
        if y_pred.shape != (h,):
            raise ValueError(
                f"model_fn returned shape {y_pred.shape}, expected ({h},)."
            )

        y_true_arr = test.values.astype(float)
        metrics = score(y_true_arr, y_pred)
        rows.append(
            {
                "fold": k,
                "cut_time": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "train_len": len(train),
                **metrics,
            }
        )
        forecasts[k] = pd.Series(y_pred, index=test.index)

        if verbose:
            print(
                f"  fold {k:>2}: cut={train.index[-1]}  train_len={len(train):>5} | "
                f"MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
                f"MAPE={metrics['MAPE_%']:.2f}%"
            )

    per_fold = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    metric_cols = ["MAE", "RMSE", "MAPE_%", "sMAPE_%"]
    aggregate = pd.DataFrame(
        {
            "mean": per_fold[metric_cols].mean(),
            "median": per_fold[metric_cols].median(),
            "std": per_fold[metric_cols].std(ddof=0),
        }
    )
    return per_fold, aggregate, forecasts


def plot_per_fold_mae_rmse(
    per_fold_mae: pd.DataFrame,
    per_fold_rmse: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (12, 4),
    y_label: str = "MW",
) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    per_fold_mae.plot(ax=axes[0], marker="o")
    axes[0].set_title("Per-fold MAE")
    axes[0].set_xlabel("fold (0 = most recent)")
    axes[0].set_ylabel(y_label)
    axes[0].grid(alpha=0.3)

    per_fold_rmse.plot(ax=axes[1], marker="o")
    axes[1].set_title("Per-fold RMSE")
    axes[1].set_xlabel("fold (0 = most recent)")
    axes[1].set_ylabel(y_label)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_sarima_prediction_interval_fold(
    y: pd.Series,
    *,
    test: pd.Series,
    median: pd.Series,
    lo95: pd.Series,
    hi95: pd.Series,
    lo80: pd.Series,
    hi80: pd.Series,
    fold: int = 0,
    ctx_hours: int = 48,
    y_label: str = "MW",
    figsize: tuple[float, float] = (11, 4.5),
) -> tuple[Figure, Axes]:
    ctx_start = test.index[0] - pd.Timedelta(hours=ctx_hours)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y.loc[ctx_start : test.index[-1]], color="black", alpha=0.8, label="actual")
    ax.plot(median.index, median.values, "--", color="tab:purple", label="SARIMA median (q=0.5)")
    ax.fill_between(test.index, lo95, hi95, color="tab:purple", alpha=0.12, label="95% PI")
    ax.fill_between(
        test.index,
        lo80,
        hi80,
        color="tab:purple",
        alpha=0.28,
        label="80% PI (q=0.1–0.9)",
    )
    ax.axvline(test.index[0], color="gray", linestyle=":", alpha=0.7, label="fold cut")
    ax.set_title(
        f"Fold {fold}: SARIMA point forecast with 80% and 95% prediction intervals"
    )
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def plot_fold_point_forecasts_overlay(
    y: pd.Series,
    *,
    seasonal_naive: pd.Series,
    ets: pd.Series,
    sarima: pd.Series,
    fold: int = 0,
    horizon_h: int,
    ctx_hours: int = 72,
    y_label: str = "MW",
    figsize: tuple[float, float] = (11, 4),
) -> tuple[Figure, Axes]:
    test_idx = seasonal_naive.index
    ctx_start = test_idx[0] - pd.Timedelta(hours=ctx_hours)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y.loc[ctx_start : test_idx[-1]], color="black", alpha=0.6, label="actual")
    ax.plot(seasonal_naive, "--", marker="o", markersize=3, label="Seasonal naive")
    ax.plot(ets, "--", marker="o", markersize=3, label="ETS")
    ax.plot(sarima, "--", marker="o", markersize=3, label="SARIMA")
    ax.axvline(test_idx[0], color="gray", linestyle=":", alpha=0.7, label="fold cut")
    ax.set_title(
        f"Fold {fold} — {horizon_h}h horizon, last {ctx_hours}h of train shown for context"
    )
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax
