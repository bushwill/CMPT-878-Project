import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Constants
GENOTYPE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                   '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
TREATMENT_COLORS = {'Control': '#2ca02c', 'DR_100': '#d62728'}
SAMPLE_PATTERN = r'(Camelina\d+)'


def _load_randomization_maps(randomization_path: str, experiment_num: int) -> tuple[dict, dict]:
    """Load treatment and genotype mappings from randomization file.
    
    Returns:
        Tuple of (treatment_map, genotype_map) dictionaries
    """
    randomization = pd.read_excel(randomization_path)
    experiment_name = f"CamelinaMAGIC_{experiment_num}.0"
    exp_randomization = randomization[randomization['Experiment'] == experiment_name]
    
    treatment_map = dict(zip(exp_randomization['Name'], exp_randomization['Treatment']))
    genotype_map = dict(zip(exp_randomization['Name'], exp_randomization['Genotype']))
    
    return treatment_map, genotype_map


def _read_and_process_timeseries_csv(
    csv_path: str,
    randomization_path: str,
    experiment_num: int,
    value_column_name: str,
    resample_freq: str = None
) -> pd.DataFrame:
    """Common logic for reading weight and transpiration CSVs.
    
    Args:
        csv_path: Path to CSV file
        randomization_path: Path to randomization Excel file
        experiment_num: Experiment number (1, 2, or 3)
        value_column_name: Name for the value column in output ('Weight' or 'Transpiration')
        resample_freq: Optional resampling frequency (e.g., "2H")
    
    Returns:
        Processed DataFrame in long format
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(randomization_path):
        raise FileNotFoundError(f"Randomization file not found: {randomization_path}")
    
    df = pd.read_csv(csv_path)
    if "Timestamp" not in df.columns:
        raise ValueError("Expected a 'Timestamp' column in the CSV")
    
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    
    if resample_freq:
        df = df.resample(resample_freq, label="left", closed="left").mean()
    
    days_since_start = (df.index - df.index.min()).total_seconds() / (24 * 3600)
    df["Days"] = days_since_start
    df = df.reset_index(drop=True)
    
    treatment_map, genotype_map = _load_randomization_maps(randomization_path, experiment_num)
    
    df_long = pd.melt(
        df,
        id_vars=['Days'],
        value_vars=[col for col in df.columns if col != 'Days'],
        var_name='Sample_Column',
        value_name=value_column_name
    )
    
    df_long['Sample'] = df_long['Sample_Column'].str.extract(SAMPLE_PATTERN)[0]
    df_long['Experiment'] = experiment_num
    df_long['Treatment'] = df_long['Sample'].map(treatment_map)
    df_long['Genotype'] = df_long['Sample'].map(genotype_map)
    
    return df_long[['Days', 'Sample', value_column_name, 'Experiment', 'Treatment', 'Genotype']].dropna(subset=[value_column_name])


def read_weight_csv(weight_csv_path: str, randomization_path: str, experiment_num: int, resample_freq: str = "2H") -> pd.DataFrame:
    """Read a raw weight CSV and enrich it with experiment and treatment information.

    This function:
      1. Reads the raw weight CSV (one experiment at a time)
      2. Reads the randomization file to get treatment assignments
      3. Resamples weight measurements to regular intervals (default 2 hours)
      4. Transforms the data from wide format to long format
      5. Adds 'Experiment', 'Sample', 'Treatment', and 'Days' columns

    Args:
      weight_csv_path: Path to a single weight CSV file (e.g., CamelinaMAGIC1.0-WeightRaw.csv)
      randomization_path: Path to the randomization Excel file
      experiment_num: Experiment number (1, 2, or 3)
      resample_freq: Resample frequency (default "2H" for 2 hours)

    Returns:
      A pandas DataFrame with columns:
        - Days: Days since experiment start
        - Sample: Sample name (e.g., 'Camelina01')
        - Weight: Weight in grams
        - Experiment: Experiment number (1, 2, or 3)
        - Treatment: 'Control' or 'DR_100'
        - Genotype: Genotype name (e.g., 'Hoga', 'Blaine Creek', etc.)
    """
    return _read_and_process_timeseries_csv(
        weight_csv_path, randomization_path, experiment_num, 'Weight', resample_freq
    )


def _calculate_stats_with_ci(data: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Calculate mean, std, and 95% confidence interval for grouped data.
    
    Args:
        data: DataFrame to process
        group_col: Column to group by (e.g., 'Days')
        value_col: Column to calculate statistics for
    
    Returns:
        DataFrame with mean, lower, and upper confidence bounds
    """
    stats = data.groupby(group_col)[value_col].agg(['mean', 'std', 'count']).reset_index()
    # CI = mean ± 1.96 * (std / sqrt(n))
    stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
    stats['lower'] = stats['mean'] - stats['ci']
    stats['upper'] = stats['mean'] + stats['ci']
    return stats


def _plot_by_genotype(
    df: pd.DataFrame,
    treatment: str,
    value_col: str,
    ylabel: str,
    title_prefix: str,
    figsize: tuple,
    title: str = None,
    save_path: str = None
) -> tuple:
    """Common plotting logic for genotype-based plots.
    
    Args:
        df: Input DataFrame
        treatment: Treatment to filter ('Control' or 'DR_100')
        value_col: Column name for values to plot ('Weight' or 'Transpiration')
        ylabel: Y-axis label
        title_prefix: Prefix for auto-generated title
        figsize: Figure size tuple
        title: Optional custom title
        save_path: Optional path to save figure
    
    Returns:
        Tuple of (fig, ax)
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    df_filtered = df[df['Treatment'] == treatment].copy()
    
    if df_filtered.empty:
        treatment_name = 'control' if treatment == 'Control' else 'drought'
        raise ValueError(f"No {treatment_name} samples found in the DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    genotypes = sorted(df_filtered['Genotype'].unique())
    colors = [GENOTYPE_COLORS[i % len(GENOTYPE_COLORS)] for i in range(len(genotypes))]
    
    for i, genotype in enumerate(genotypes):
        genotype_data = df_filtered[df_filtered['Genotype'] == genotype]
        stats = _calculate_stats_with_ci(genotype_data, 'Days', value_col)
        
        ax.plot(stats['Days'], stats['mean'], label=genotype, color=colors[i], linewidth=2)
        ax.fill_between(stats['Days'], stats['lower'], stats['upper'], color=colors[i], alpha=0.2)
    
    ax.set_xlabel("Days since experiment start")
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', title='Genotype')
    
    if title:
        ax.set_title(title)
    else:
        exp_num = df_filtered['Experiment'].iloc[0]
        ax.set_title(f"{title_prefix} - Experiment {exp_num}")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    return fig, ax


def plot_drought_weight_df(df: pd.DataFrame, figsize=(12, 6), title=None, save_path=None):
    """Plot weight measurements for drought samples grouped by genotype.
    
    Shows mean weight with 95% confidence interval (shaded area) for each genotype.

    Args:
      df: DataFrame from read_weight_csv (should contain one experiment)
      figsize: Figure size (width, height)
      title: Optional plot title
      save_path: Optional path to save the figure

    Returns:
      (fig, ax) tuple of the created Matplotlib objects.
    """
    return _plot_by_genotype(
        df, 'DR_100', 'Weight', 'Weight (g)',
        'Drought Samples by Genotype', figsize, title, save_path
    )


def plot_control_weight_df(df: pd.DataFrame, figsize=(12, 6), title=None, save_path=None):
    """Plot weight measurements for control samples grouped by genotype.
    
    Shows mean weight with 95% confidence interval (shaded area) for each genotype.

    Args:
      df: DataFrame from read_weight_csv (should contain one experiment)
      figsize: Figure size (width, height)
      title: Optional plot title
      save_path: Optional path to save the figure

    Returns:
      (fig, ax) tuple of the created Matplotlib objects.
    """
    return _plot_by_genotype(
        df, 'Control', 'Weight', 'Weight (g)',
        'Control Samples by Genotype', figsize, title, save_path
    )


def read_transpiration_csv(transpiration_csv_path: str, randomization_path: str, experiment_num: int) -> pd.DataFrame:
    """Read a daily transpiration CSV and enrich it with experiment and treatment information.

    This function:
      1. Reads the daily transpiration CSV (one experiment at a time)
      2. Reads the randomization file to get treatment assignments
      3. Transforms the data from wide format to long format
      4. Adds 'Experiment', 'Sample', 'Treatment', 'Genotype', and 'Days' columns

    Args:
      transpiration_csv_path: Path to a single transpiration CSV file (e.g., CamelinaMAGIC1.0-DailyTranspiration.csv)
      randomization_path: Path to the randomization Excel file
      experiment_num: Experiment number (1, 2, or 3)

    Returns:
      A pandas DataFrame with columns:
        - Days: Days since experiment start
        - Sample: Sample name (e.g., 'Camelina01')
        - Transpiration: Daily transpiration in grams
        - Experiment: Experiment number (1, 2, or 3)
        - Treatment: 'Control' or 'DR_100'
        - Genotype: Genotype name (e.g., 'Hoga', 'Blaine Creek', etc.)
    """
    return _read_and_process_timeseries_csv(
        transpiration_csv_path, randomization_path, experiment_num, 'Transpiration'
    )


def plot_drought_transpiration_df(df: pd.DataFrame, figsize=(12, 6), title=None, save_path=None):
    """Plot daily transpiration for drought samples grouped by genotype.
    
    Shows mean transpiration with 95% confidence interval (shaded area) for each genotype.

    Args:
      df: DataFrame from read_transpiration_csv (should contain one experiment)
      figsize: Figure size (width, height)
      title: Optional plot title
      save_path: Optional path to save the figure

    Returns:
      (fig, ax) tuple of the created Matplotlib objects.
    """
    return _plot_by_genotype(
        df, 'DR_100', 'Transpiration', 'Daily Transpiration (g)',
        'Drought Samples - Daily Transpiration by Genotype', figsize, title, save_path
    )


def plot_control_transpiration_df(df: pd.DataFrame, figsize=(12, 6), title=None, save_path=None):
    """Plot daily transpiration for control samples grouped by genotype.
    
    Shows mean transpiration with 95% confidence interval (shaded area) for each genotype.

    Args:
      df: DataFrame from read_transpiration_csv (should contain one experiment)
      figsize: Figure size (width, height)
      title: Optional plot title
      save_path: Optional path to save the figure

    Returns:
      (fig, ax) tuple of the created Matplotlib objects.
    """
    return _plot_by_genotype(
        df, 'Control', 'Transpiration', 'Daily Transpiration (g)',
        'Control Samples - Daily Transpiration by Genotype', figsize, title, save_path
    )


def _plot_by_treatment(
    df: pd.DataFrame,
    genotype: str,
    value_col: str,
    ylabel: str,
    title_template: str,
    figsize: tuple,
    title: str = None,
    save_path: str = None
) -> tuple:
    """Common plotting logic for treatment comparison plots.
    
    Args:
        df: Input DataFrame
        genotype: Genotype name to filter
        value_col: Column name for values to plot
        ylabel: Y-axis label
        title_template: Template string for auto title (formatted with genotype and exp_num)
        figsize: Figure size tuple
        title: Optional custom title
        save_path: Optional path to save figure
    
    Returns:
        Tuple of (fig, ax)
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    df_genotype = df[df['Genotype'] == genotype].copy()
    
    if df_genotype.empty:
        raise ValueError(f"No data found for genotype '{genotype}' in the DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    treatments = sorted(df_genotype['Treatment'].unique())
    
    for treatment in treatments:
        treatment_data = df_genotype[df_genotype['Treatment'] == treatment]
        stats = _calculate_stats_with_ci(treatment_data, 'Days', value_col)
        
        color = TREATMENT_COLORS.get(treatment, '#1f77b4')
        label = 'Control' if treatment == 'Control' else 'Drought'
        
        ax.plot(stats['Days'], stats['mean'], label=label, color=color, linewidth=2)
        ax.fill_between(stats['Days'], stats['lower'], stats['upper'], color=color, alpha=0.2)
    
    ax.set_xlabel("Days since experiment start")
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', title='Treatment')
    
    if title:
        ax.set_title(title)
    else:
        exp_num = df_genotype['Experiment'].iloc[0]
        ax.set_title(title_template.format(genotype=genotype, exp_num=exp_num))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    return fig, ax


def plot_genotype_weight_df(df: pd.DataFrame, genotype: str, figsize=(12, 6), title=None, save_path=None):
    """Plot weight measurements for a specific genotype in both control and drought treatments.
    
    Shows mean weight with 95% confidence interval (shaded area) for both treatments.

    Args:
      df: DataFrame from read_weight_csv (should contain one experiment)
      genotype: Genotype name to plot (e.g., 'Hoga', 'Blaine Creek', etc.)
      figsize: Figure size (width, height)
      title: Optional plot title
      save_path: Optional path to save the figure

    Returns:
      (fig, ax) tuple of the created Matplotlib objects.
    """
    return _plot_by_treatment(
        df, genotype, 'Weight', 'Weight (g)',
        'Weight for {genotype} - Control vs Drought - Experiment {exp_num}',
        figsize, title, save_path
    )


def plot_genotype_transpiration_df(df: pd.DataFrame, genotype: str, figsize=(12, 6), title=None, save_path=None):
    """Plot daily transpiration for a specific genotype in both control and drought treatments.
    
    Shows mean transpiration with 95% confidence interval (shaded area) for both treatments.

    Args:
      df: DataFrame from read_transpiration_csv (should contain one experiment)
      genotype: Genotype name to plot (e.g., 'Hoga', 'Blaine Creek', etc.)
      figsize: Figure size (width, height)
      title: Optional plot title
      save_path: Optional path to save the figure

    Returns:
      (fig, ax) tuple of the created Matplotlib objects.
    """
    return _plot_by_treatment(
        df, genotype, 'Transpiration', 'Daily Transpiration (g)',
        'Daily Transpiration for {genotype} - Control vs Drought - Experiment {exp_num}',
        figsize, title, save_path
    )


def read_weather_station_csv(weather_csv_path: str, experiment_num: int, resample_freq: str = "2H") -> pd.DataFrame:
    """Read a weather station CSV and resample to regular intervals.

    This function:
      1. Reads the weather station CSV (one experiment at a time)
      2. Resamples weather measurements to regular intervals (default 2 hours)
      3. Calculates mean values for each interval
      4. Adds 'Days' column for days since experiment start

    Args:
      weather_csv_path: Path to a single weather station CSV file (e.g., CamelinaMAGIC1.0-WeatherStation.csv)
      experiment_num: Experiment number (1, 2, or 3)
      resample_freq: Resample frequency (default "2H" for 2 hours)

    Returns:
      A pandas DataFrame with columns:
        - Days: Days since experiment start
        - PARLight: Photosynthetically Active Radiation (µmol/m²/s)
        - RH: Relative Humidity (%)
        - Temp: Temperature (°C)
        - VPD: Vapor Pressure Deficit (kPa)
        - Experiment: Experiment number (1, 2, or 3)
    """
    if not os.path.exists(weather_csv_path):
        raise FileNotFoundError(f"Weather station CSV not found: {weather_csv_path}")
    
    df_weather = pd.read_csv(weather_csv_path)
    if "Timestamp" not in df_weather.columns:
        raise ValueError("Expected a 'Timestamp' column in the weather CSV")
    
    df_weather["Timestamp"] = pd.to_datetime(df_weather["Timestamp"])
    df_weather = df_weather.set_index("Timestamp").sort_index()
    
    column_mapping = {}
    for col in df_weather.columns:
        if 'PARLight' in col:
            column_mapping[col] = 'PARLight'
        elif 'RH' in col:
            column_mapping[col] = 'RH'
        elif 'Temp' in col:
            column_mapping[col] = 'Temp'
        elif 'VPD' in col:
            column_mapping[col] = 'VPD'
    
    df_weather = df_weather.rename(columns=column_mapping)
    df_weather = df_weather.apply(pd.to_numeric, errors="coerce")
    df_resampled = df_weather.resample(resample_freq, label="left", closed="left").mean()
    
    days_since_start = (df_resampled.index - df_resampled.index.min()).total_seconds() / (24 * 3600)
    df_resampled["Days"] = days_since_start
    df_resampled['Experiment'] = experiment_num
    df_resampled = df_resampled.reset_index(drop=True)
    df_resampled = df_resampled[['Days', 'PARLight', 'RH', 'Temp', 'VPD', 'Experiment']]
    df_resampled = df_resampled.dropna(how='all', subset=['PARLight', 'RH', 'Temp', 'VPD'])
    
    return df_resampled


def plot_weather_station_data(df: pd.DataFrame, figsize=(12, 10), title=None, save_path=None):
    """Plot weather station data with multiple subplots for each climate variable.
    
    Creates a 4-panel plot showing:
      - PAR Light (Photosynthetically Active Radiation)
      - Relative Humidity
      - Temperature
      - Vapor Pressure Deficit

    Args:
      df: DataFrame from read_weather_station_csv (should contain one experiment)
      figsize: Figure size (width, height)
      title: Optional plot title
      save_path: Optional path to save the figure

    Returns:
      (fig, axes) tuple of the created Matplotlib objects.
    """
    
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    axes[0].plot(df['Days'], df['PARLight'], color='#ff7f0e', linewidth=1)
    axes[0].set_ylabel('PAR Light\n(µmol/m²/s)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df['Days'], df['RH'], color='#2ca02c', linewidth=1)
    axes[1].set_ylabel('Relative Humidity\n(%)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(df['Days'], df['Temp'], color='#d62728', linewidth=1)
    axes[2].set_ylabel('Temperature\n(°C)')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(df['Days'], df['VPD'], color='#9467bd', linewidth=1)
    axes[3].set_ylabel('VPD\n(kPa)')
    axes[3].set_xlabel('Days since experiment start')
    axes[3].grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        exp_num = df['Experiment'].iloc[0]
        fig.suptitle(f"Weather Station Data - Experiment {exp_num}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    return fig, axes


def plot_weather_station_data_enhanced(dataframes_dict, show_debug=False):
    """Plot enhanced weather station data with daily min/max/mean statistics.
    
    Creates two types of plots for each experiment:
      1. Combined overlay plot showing all climate variables
      2. Individual subplot grid with shaded min/max regions
    
    Args:
      dataframes_dict: Dictionary mapping experiment names to weather DataFrames
                      e.g., {"Experiment 1": weather_df_1, "Experiment 2": weather_df_2}
      show_debug: If True, prints debug information during processing
    
    Returns:
      None (displays plots)
    """
    
    for exp_name, df in dataframes_dict.items():
        if df is None or df.empty:
            print(f"Skipping {exp_name}: DataFrame is empty or None")
            continue
        
        if show_debug:
            print(f"\n{'='*60}")
            print(f"Processing: {exp_name}")
            print(f"{'='*60}")
        
        days_col = [col for col in df.columns if 'day' in col.lower()]
        if not days_col:
            print(f"Warning: No 'Days' column found in {exp_name}")
            continue
        days_col = days_col[0]
        
        if show_debug:
            print(f"Days column: {days_col}")
        
        climate_vars = [col for col in df.select_dtypes(include=[np.number]).columns if col != days_col and col != 'Experiment']        
        if show_debug:
            print(f"Climate variables found: {climate_vars}")
        
        df['day_integer'] = np.floor(df[days_col]).astype(int)
        daily_stats = df.groupby('day_integer')[climate_vars].agg(['min', 'max', 'mean'])
        
        if show_debug:
            print(f"\nDaily statistics shape: {daily_stats.shape}")
            print(f"Day range: {daily_stats.index.min()} to {daily_stats.index.max()}")
        
        # Combined overlay plot
        _, ax1 = plt.subplots(figsize=(14, 6))
        
        colors = sns.color_palette("husl", len(climate_vars))
        
        for i, var in enumerate(climate_vars):
            ax1.plot(daily_stats.index, daily_stats[(var, 'min')], 
                    label=f'{var} (min)', color=colors[i], linestyle='--', alpha=0.6)
            ax1.plot(daily_stats.index, daily_stats[(var, 'max')], 
                    label=f'{var} (max)', color=colors[i], linestyle=':', alpha=0.6)
            ax1.plot(daily_stats.index, daily_stats[(var, 'mean')], 
                    label=f'{var} (mean)', color=colors[i], linewidth=2)
        
        ax1.set_xlabel('Day (integer)')
        ax1.set_ylabel('Climate Variable Values')
        ax1.set_title(f'{exp_name} - All Climate Variables (Daily Min/Max/Mean)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Individual subplots with shaded regions
        n_vars = len(climate_vars)
        fig2, axes = plt.subplots(n_vars, 1, figsize=(14, 3*n_vars), sharex=True)
        
        if n_vars == 1:
            axes = [axes]
        
        for i, var in enumerate(climate_vars):
            ax = axes[i]
            
            ax.plot(daily_stats.index, daily_stats[(var, 'mean')], 
                   label='Mean', color='blue', linewidth=2)
            
            ax.fill_between(daily_stats.index, 
                           daily_stats[(var, 'min')], 
                           daily_stats[(var, 'max')],
                           alpha=0.3, color='lightblue', label='Min-Max Range')
            
            ax.plot(daily_stats.index, daily_stats[(var, 'min')], 
                   label='Min', color='blue', linestyle='--', alpha=0.6)
            ax.plot(daily_stats.index, daily_stats[(var, 'max')], 
                   label='Max', color='blue', linestyle=':', alpha=0.6)
            
            ax.set_ylabel(var)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Day (integer)')
        fig2.suptitle(f'{exp_name} - Daily Min/Max/Mean for Each Climate Variable', 
                     fontsize=14, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.show()
        
        if show_debug:
            print(f"\nCompleted plots for {exp_name}")
            print(f"{'='*60}\n")


def get_random_samples_by_genotype(transpiration_dfs, genotypes):
    """Select one random sample for each genotype across all experiments."""
    all_data = pd.concat(transpiration_dfs)
    results = {}
    
    for genotype in genotypes:
        samples = all_data[all_data['Genotype'] == genotype]['Sample'].unique()
        if len(samples) > 0:
            sample_name = np.random.choice(samples)
            sample_data = all_data[all_data['Sample'] == sample_name].sort_values('Days')
            results[genotype] = (sample_name, sample_data)
    
    return results


def calculate_mean_daily_change(data, start_day, end_day, column='Transpiration_Pct'):
    """Calculate mean daily change in transpiration for a given period.
    
    Args:
        data: DataFrame with transpiration data
        start_day: Start day of the period
        end_day: End day of the period
        column: Column to calculate changes from (default 'Transpiration_Pct')
    
    Returns:
        Tuple of (mean_change, total_change, days_in_period)
    """
    period_data = data[(data['Days'] >= start_day) & (data['Days'] <= end_day)].sort_values('Days')
    
    if len(period_data) < 2:
        return None, None, None
    
    daily_changes = period_data[column].diff().dropna()
    mean_change = daily_changes.mean()
    total_change = period_data[column].iloc[-1] - period_data[column].iloc[0]
    days_in_period = period_data['Days'].iloc[-1] - period_data['Days'].iloc[0]
    
    return mean_change, total_change, days_in_period


def analyze_sample_periods(sample_data, sample_name, periods, use_percent=True):
    """Analyze transpiration changes across multiple time periods.
    
    Args:
        sample_data: DataFrame with transpiration data
        sample_name: Name of the sample
        periods: List of tuples (period_name, start_day, end_day)
        use_percent: If True, calculate based on % of raw weight; if False, use g/day
    
    Returns:
        Dictionary with analysis results
    """
    column = 'Transpiration_Pct' if use_percent else 'Transpiration'
    unit = '% of raw weight' if use_percent else 'g'
    
    results = {
        'sample': sample_name,
        'experiment': sample_data['Experiment'].iloc[0],
        'treatment': sample_data['Treatment'].iloc[0],
        'genotype': sample_data['Genotype'].iloc[0],
        'total_transpiration': sample_data['Transpiration'].sum(),
        'mean_daily_transpiration': sample_data['Transpiration'].mean(),
        'unit': unit,
        'periods': {}
    }
    
    for period_name, start_day, end_day in periods:
        mean_change, total_change, days = calculate_mean_daily_change(sample_data, start_day, end_day, column)
        results['periods'][period_name] = {
            'mean_daily_change': mean_change,
            'total_change': total_change,
            'days': days
        }
    
    return results


def plot_sample_comparison(sample_dict, figsize=(12, 6)):
    """Plot transpiration comparison for multiple samples."""
    fig, ax = plt.subplots(figsize=figsize)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, (genotype, (sample_name, data)) in enumerate(sample_dict.items()):
        ax.plot(data['Days'], data['Transpiration'], 
                label=f'{genotype} ({sample_name}, {data["Treatment"].iloc[0]})',
                linewidth=2, marker=markers[idx % len(markers)], markersize=4)
    
    ax.set_xlabel('Days since experiment start', fontsize=12)
    ax.set_ylabel('Daily Transpiration (g)', fontsize=12)
    ax.set_title('Transpiration Comparison: Individual Samples', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


def print_sample_analysis(analysis_results):
    """Print formatted analysis results for a sample."""
    unit = analysis_results.get('unit', 'g')
    
    print(f"\n{analysis_results['genotype']} Sample ({analysis_results['sample']}):")
    print("-" * 80)
    print(f"Experiment: {analysis_results['experiment']}")
    print(f"Treatment: {analysis_results['treatment']}")
    print(f"Total transpiration: {analysis_results['total_transpiration']:.2f} g")
    print(f"Mean daily transpiration: {analysis_results['mean_daily_transpiration']:.2f} g")
    
    for period_name, period_data in analysis_results['periods'].items():
        print(f"\n{period_name}:")
        if period_data['mean_daily_change'] is not None:
            print(f"  Mean daily change: {period_data['mean_daily_change']:.2f} {unit}/day")
            print(f"  Total change: {period_data['total_change']:.2f} {unit} over {period_data['days']:.1f} days")
        else:
            print("  Insufficient data")
