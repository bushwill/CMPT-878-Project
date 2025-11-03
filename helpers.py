import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt



def read_weight_csv(path: str, resample_freq: str = "2H") -> pd.DataFrame:
    """Read a raw weight CSV produced by the lysimeter system and return a
    DataFrame resampled to a regular interval (default 2 hours).

    The CSV is expected to have a `Timestamp` column (ISO format) and one
    column per sample like `Camelina01 - Weight (g)`. The function will:
      - read the file with pandas
      - parse `Timestamp` into a datetime index
      - sort the index
      - resample to `resample_freq` (default 2 hours) using the mean of the
        measurements within each window

    Returns:
      A pandas DataFrame with the resampled time index and the same sample
      columns (weights in grams). For a 24-hour day and the default `2H`
      frequency there will be 12 values per sample per day.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight CSV not found: {path}")

    # Read CSV and parse timestamps
    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        raise ValueError("Expected a 'Timestamp' column in the weight CSV")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp").sort_index()

    # If there are non-numeric columns besides the timestamp, try to coerce
    # sample columns to numeric (this will convert empty/invalid to NaN).
    df = df.apply(pd.to_numeric, errors="coerce")

    # Resample to the requested frequency using mean aggregation. This will
    # produce one value per interval (e.g. 12 values per day for '2H').
    resampled = df.resample(resample_freq, label="left", closed="left").mean()

    return resampled


def plot_weight_df(df: pd.DataFrame, figsize=(12, 6), title: Optional[str] = None, save_path: Optional[str] = None):
    """Plot the resampled weight DataFrame using matplotlib.

    - Plots each sample column as a separate line.
    - Hides the legend (per request).
    - Labels x/y axes and optional title.
    - If save_path is provided, saves the figure to that path.

    Returns:
      (fig, ax) tuple of the created Matplotlib objects.
    """

    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    fig, ax = plt.subplots(figsize=figsize)

    # Use pandas plotting for convenience, but disable the legend.
    df.plot(ax=ax, legend=False)

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Weight (g)")
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax