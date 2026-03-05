"""HPLC peak detection and area integration for headerless CSV-like text files.

This module provides reusable functions to:
1) load malformed comma-separated traces safely,
2) detect the main chromatographic peak,
3) integrate area under the peak without baseline zeroing,
4) visualize trace with shaded integrated region.

Dependencies are limited to numpy, pandas, and matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PeakRegion:
    """Container for detected peak boundaries and apex index."""

    start_idx: int
    apex_idx: int
    end_idx: int


class HPLCDataError(ValueError):
    """Raised when the input file cannot be parsed into a usable chromatogram."""


def load_hplc_trace(file_path: str | Path, min_rows: int = 10) -> pd.DataFrame:
    """Load a headerless HPLC text file robustly.

    Assumes comma-separated values with no header and at least two columns:
    x-axis (time) and y-axis (signal). Malformed lines are skipped.

    Parameters
    ----------
    file_path:
        Path to the .txt data file.
    min_rows:
        Minimum required valid data points after cleaning.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['x', 'y'] sorted by x.

    Raises
    ------
    HPLCDataError
        If file cannot be read or cleaned into a valid trace.
    """
    try:
        raw = pd.read_csv(
            file_path,
            header=None,
            sep=",",
            engine="python",
            on_bad_lines="skip",
            comment="#",
        )
    except Exception as exc:
        raise HPLCDataError(f"Unable to read file '{file_path}': {exc}") from exc

    if raw.shape[1] < 2:
        raise HPLCDataError(
            "Input file must contain at least two comma-separated columns (x, y)."
        )

    trace = raw.iloc[:, :2].copy()
    trace.columns = ["x", "y"]

    # Force numeric conversion and remove malformed rows.
    trace["x"] = pd.to_numeric(trace["x"], errors="coerce")
    trace["y"] = pd.to_numeric(trace["y"], errors="coerce")
    trace = trace.dropna(subset=["x", "y"])

    # Drop duplicate x-values by averaging to keep monotonic integration grid stable.
    trace = trace.groupby("x", as_index=False)["y"].mean().sort_values("x")

    if len(trace) < min_rows:
        raise HPLCDataError(
            f"Not enough valid numeric points after cleaning (found {len(trace)})."
        )

    if np.isclose(trace["x"].diff().fillna(1.0).abs().sum(), 0.0):
        raise HPLCDataError("x-axis values appear constant; cannot integrate over a flat axis.")

    return trace.reset_index(drop=True)


def _smooth_signal(y: np.ndarray, window: int = 11) -> np.ndarray:
    """Simple moving-average smoothing with odd window length."""
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window >= len(y):
        window = max(3, len(y) // 2 * 2 - 1)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def detect_main_peak(
    x: np.ndarray,
    y: np.ndarray,
    threshold_fraction: float = 0.02,
    smooth_window: int = 11,
) -> PeakRegion:
    """Detect the dominant peak using smoothed signal and local minima boundaries.

    Peak apex is chosen as the global maximum of the smoothed signal.
    Boundaries expand left/right from apex until the signal falls below a
    data-driven threshold or starts increasing away from local minima.
    """
    if len(x) != len(y) or len(y) < 5:
        raise HPLCDataError("x and y must be same length with at least 5 points.")

    ys = _smooth_signal(np.asarray(y, dtype=float), window=smooth_window)
    apex = int(np.nanargmax(ys))

    y_min = float(np.nanmin(ys))
    y_max = float(np.nanmax(ys))
    amplitude = y_max - y_min
    if amplitude <= 0:
        raise HPLCDataError("Signal has no positive dynamic range; peak not detectable.")

    level = y_min + threshold_fraction * amplitude

    left = apex
    while left > 0:
        if ys[left] <= level:
            break
        if ys[left - 1] > ys[left] and ys[left] < ys[apex]:
            break
        left -= 1

    right = apex
    n = len(ys)
    while right < n - 1:
        if ys[right] <= level:
            break
        if ys[right + 1] > ys[right] and ys[right] < ys[apex]:
            break
        right += 1

    if left >= apex or right <= apex:
        # Conservative fallback around apex to ensure non-empty region.
        half_width = max(2, n // 50)
        left = max(0, apex - half_width)
        right = min(n - 1, apex + half_width)

    return PeakRegion(start_idx=left, apex_idx=apex, end_idx=right)


def integrate_peak_area(x: np.ndarray, y: np.ndarray, region: PeakRegion) -> float:
    """Integrate peak area over the detected region without baseline subtraction."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    sl = slice(region.start_idx, region.end_idx + 1)
    return float(np.trapz(ys[sl], xs[sl]))


def plot_peak_integration(
    x: np.ndarray,
    y: np.ndarray,
    region: PeakRegion,
    area: float,
    title: str = "HPLC Main Peak Integration",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot chromatogram and shade the integrated peak region."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(xs, ys, lw=1.5, color="tab:blue", label="Signal")

    sl = slice(region.start_idx, region.end_idx + 1)
    ax.fill_between(xs[sl], ys[sl], alpha=0.35, color="tab:orange", label="Integrated area")
    ax.axvline(xs[region.apex_idx], color="tab:red", ls="--", lw=1, label="Peak apex")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.legend(loc="best")
    ax.text(
        0.99,
        0.95,
        f"Area = {area:.6g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    return fig, ax


def analyze_hplc_file(file_path: str | Path) -> Tuple[pd.DataFrame, PeakRegion, float]:
    """End-to-end reusable analysis pipeline for a single file."""
    trace = load_hplc_trace(file_path)
    region = detect_main_peak(trace["x"].to_numpy(), trace["y"].to_numpy())
    area = integrate_peak_area(trace["x"].to_numpy(), trace["y"].to_numpy(), region)
    return trace, region, area


def main(file_path: str | Path) -> None:
    """Programmatic entrypoint for analyzing a single HPLC file path."""
    trace, region, area = analyze_hplc_file(file_path)
    print(
        "Detected main peak:",
        f"start={trace['x'].iloc[region.start_idx]:.6g},",
        f"apex={trace['x'].iloc[region.apex_idx]:.6g},",
        f"end={trace['x'].iloc[region.end_idx]:.6g}",
    )
    print(f"Integrated area (no baseline zeroing): {area:.10g}")

    plot_peak_integration(trace["x"].to_numpy(), trace["y"].to_numpy(), region, area)
    plt.show()


def cli_main() -> None:
    """Command-line entrypoint for ``python -m hplc_peak_integration`` or console script usage."""
    import argparse

    parser = argparse.ArgumentParser(description="HPLC peak integration from headerless .txt")
    parser.add_argument("file", type=str, help="Path to comma-separated .txt file (no header)")
    args = parser.parse_args()

    try:
        main(args.file)
    except HPLCDataError as err:
        print(f"[ERROR] {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    cli_main()
