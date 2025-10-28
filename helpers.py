"""Helper functions for reading Camelina MAGIC datasets and plotting visualizations.

Functions:
- read_biomass_workbook(path): auto-detects and returns the biomass sheet DataFrame
- get_start_weights_from_weightraw(path): returns DataFrame with Pot and StartWeight
- build_biomass_dataframe(biomass_df, start_weights): returns tidy DataFrame with Pot, StartWeight, FreshShoot, FreshRoot, FreshTotal
- plot_parallel_coordinates(df, outpath, normalize=True): saves a parallel-coordinates plot

The functions are defensive and print detected column mappings so you can confirm them
when running in the notebook.
"""
from typing import Tuple, Dict, List
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first column in df whose lowercased name contains any candidate substring."""
    cols = df.columns
    lower = [c.lower() for c in cols]
    for cand in candidates:
        lc = cand.lower()
        for i, name in enumerate(lower):
            if lc in name:
                return cols[i]
    return None


def read_biomass_workbook(path: str) -> pd.DataFrame:
    """Open the biomass workbook and return the most likely sheet containing fresh biomass.

    The function looks for sheets that contain column names with keywords like
    'fresh' and 'shoot' or 'root' or 'total'. It prints the chosen sheet and its columns.
    """
    xls = pd.read_excel(path, sheet_name=None)
    # Candidate keywords
    want_keywords = ["fresh", "shoot", "root", "total", "biomass"]
    best_sheet = None
    best_score = -1
    for sheet_name, df in xls.items():
        joined = " ".join([c.lower() for c in df.columns.astype(str)])
        score = sum(1 for k in want_keywords if k in joined)
        if score > best_score:
            best_score = score
            best_sheet = sheet_name
    if best_sheet is None:
        # fallback: return first sheet
        best_sheet = list(xls.keys())[0]
    df = xls[best_sheet]
    print(f"read_biomass_workbook: selected sheet '{best_sheet}' with {len(df.columns)} columns")
    print("columns:", list(df.columns))
    return df


def get_start_weights_from_weightraw(path: str) -> pd.DataFrame:
    """Read the weightraw CSV and return a DataFrame with columns ['Pot','StartWeight'].

    Logic: read CSV, sort by Timestamp (if present), take the earliest row and extract
    columns matching the pattern 'CamelinaNN - Weight'. NN -> pot number.
    """
    df = pd.read_csv(path)
    # canonicalize timestamp
    if 'Timestamp' in df.columns:
        try:
            df['Timestamp_parsed'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values('Timestamp_parsed')
        except Exception:
            # leave as-is
            pass
    first = df.iloc[0]
    pattern = re.compile(r'camelina\s*(\d{1,2})[^\d]*weight', flags=re.I)
    pots = []
    for col in df.columns:
        m = pattern.search(col)
        if m:
            pot = int(m.group(1))
            try:
                val = float(first[col])
            except Exception:
                val = np.nan
            pots.append((pot, val))
    if not pots:
        raise ValueError(f"No per-pot weight columns detected in {path}")
    start_df = pd.DataFrame(pots, columns=['Pot', 'StartWeight']).sort_values('Pot')
    print(f"get_start_weights_from_weightraw: found {len(start_df)} pots (sample):\n", start_df.head())
    return start_df


def build_biomass_dataframe(biomass_df: pd.DataFrame, start_weights: pd.DataFrame,
                            pot_numbers: List[int] = list(range(1, 51))) -> pd.DataFrame:
    """Build a tidy DataFrame with Pot, StartWeight, FreshShoot, FreshRoot, FreshTotal.

    Approach:
    - If biomass_df has an explicit pot/id column, use it and look up shoot/root/total columns
      by candidate name matches.
    - Otherwise, look for per-pot wide columns like 'Camelina01 - Fresh Shoot (g)'.
    """
    df = biomass_df.copy()
    # Candidate names for pot column and biomass measures
    # include 'name' because the biomass workbook uses a 'Name' column
    pot_cands = ['pot', 'pot id', 'pot_id', 'id', 'plant', 'sample', 'name']
    shoot_cands = ['fresh shoot', 'freshshoot', 'shoot (g)', 'shoot']
    root_cands = ['fresh root', 'freshroot', 'root (g)', 'root']
    total_cands = ['fresh total', 'freshtotal', 'total (g)', 'total']

    pot_col = _find_column(df, pot_cands)
    # quick helper to find measure column
    def _find_measure(candidates):
        return _find_column(df, candidates)

    if pot_col is not None:
        print(f"build_biomass_dataframe: using pot column '{pot_col}'")
        # find measure columns
        shoot_col = _find_measure(shoot_cands)
        root_col = _find_measure(root_cands)
        total_col = _find_measure(total_cands)
        print("detected columns ->", shoot_col, root_col, total_col)
        if not all([shoot_col, root_col, total_col]):
            raise ValueError("Could not detect one of the fresh shoot/root/total columns in biomass sheet.")
        sub = df[[pot_col, shoot_col, root_col, total_col]].rename(columns={pot_col: 'Pot', shoot_col: 'FreshShoot', root_col: 'FreshRoot', total_col: 'FreshTotal'})
        # ensure Pot numeric: try direct numeric, otherwise extract digits from string labels (e.g., 'Camelina01')
        sub['Pot'] = pd.to_numeric(sub['Pot'], errors='coerce')
        if sub['Pot'].isna().any():
            # attempt to extract integer from the name
            def _extract_num(x):
                try:
                    s = str(x)
                    m = re.search(r"(\d{1,3})", s)
                    return int(m.group(1)) if m else pd.NA
                except Exception:
                    return pd.NA
            sub['Pot'] = sub['Pot'].fillna(sub['Pot'].apply(lambda _: pd.NA))
            sub['Pot'] = sub['Pot'].astype(object)
            extracted = sub['Pot']
            # if original numeric failed, try extraction on original column values
            orig_vals = df[pot_col].astype(str)
            extracted_nums = orig_vals.apply(lambda v: (re.search(r"(\d{1,3})", v).group(1) if re.search(r"(\d{1,3})", v) else None))
            sub['Pot'] = pd.to_numeric(extracted_nums, errors='coerce')
        sub['Pot'] = sub['Pot'].astype('Int64')
        # merge start weights
        out = pd.merge(pd.DataFrame({'Pot': pot_numbers}), sub, on='Pot', how='left')
        out = pd.merge(out, start_weights, on='Pot', how='left')
        return out[['Pot','StartWeight','FreshShoot','FreshRoot','FreshTotal']]

    # fallback: wide per-pot columns like 'Camelina01 - Fresh Shoot (g)'
    print("build_biomass_dataframe: no explicit pot column found, attempting wide-column parsing")
    # lowercase column map
    col_map = {c: c.lower() for c in df.columns}
    # helper to find column for a given pot and measure
    def _col_for(pot, measure_keywords):
        two = f"{pot:02d}"
        for c in df.columns:
            low = c.lower()
            if two in low and any(k in low for k in measure_keywords):
                return c
        return None

    rows = []
    for pot in pot_numbers:
        shoot_col = _col_for(pot, ['shoot'])
        root_col = _col_for(pot, ['root'])
        total_col = _col_for(pot, ['total'])
        start_row = start_weights.loc[start_weights['Pot'] == pot]
        start_val = float(start_row['StartWeight'].values[0]) if len(start_row) else np.nan
        shoot_val = df[shoot_col].iloc[0] if shoot_col in df.columns else np.nan
        root_val = df[root_col].iloc[0] if root_col in df.columns else np.nan
        total_val = df[total_col].iloc[0] if total_col in df.columns else np.nan
        rows.append((pot, start_val, shoot_val, root_val, total_val))

    out = pd.DataFrame(rows, columns=['Pot','StartWeight','FreshShoot','FreshRoot','FreshTotal'])
    print("build_biomass_dataframe: assembled wide-format table (sample):\n", out.head())
    return out


def plot_parallel_coordinates(df: pd.DataFrame, outpath: str, normalize: bool = True, figsize=(10,6)) -> None:
    """Plot and save a parallel-coordinates plot from a tidy DataFrame.

    Expects df to have columns: ['Pot', 'StartWeight', 'FreshShoot', 'FreshRoot', 'FreshTotal']
    """
    needed = ['Pot','StartWeight','FreshShoot','FreshRoot','FreshTotal']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for plotting: {missing}")
    plot_df = df.copy()
    plot_df = plot_df.set_index('Pot')
    numeric_cols = ['StartWeight','FreshShoot','FreshRoot','FreshTotal']
    # convert to numeric
    for c in numeric_cols:
        plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce')
    if normalize:
        norm = (plot_df[numeric_cols] - plot_df[numeric_cols].min()) / (plot_df[numeric_cols].max() - plot_df[numeric_cols].min())
        plot_df[numeric_cols] = norm
    # prepare for parallel_coordinates: it expects the class column to be a column, so reset index
    pc_df = plot_df.reset_index()
    pc_df['PotLabel'] = pc_df['Pot'].astype(str)
    cols_for_plot = ['PotLabel'] + numeric_cols
    fig, ax = plt.subplots(figsize=figsize)
    # use pandas parallel_coordinates. don't pass a color array sized to rows (can fail),
    # allow pandas to choose the colormap. hide the legend which would be large for 50 items.
    parallel_coordinates(pc_df[cols_for_plot], 'PotLabel', linewidth=1)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_title('Parallel coordinates: start weight → fresh shoot/root/total (normalized)')
    plt.tight_layout()
    outdir = os.path.dirname(outpath) or '.'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    print(f"Saved parallel-coordinates plot to {outpath}")
