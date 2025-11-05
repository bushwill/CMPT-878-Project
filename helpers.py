import os

import pandas as pd
import matplotlib.pyplot as plt


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
    
    # Validate paths
    if not os.path.exists(weight_csv_path):
        raise FileNotFoundError(f"Weight CSV not found: {weight_csv_path}")
    if not os.path.exists(randomization_path):
        raise FileNotFoundError(f"Randomization file not found: {randomization_path}")
    
    # Read the weight CSV
    df_weight = pd.read_csv(weight_csv_path)
    if "Timestamp" not in df_weight.columns:
        raise ValueError("Expected a 'Timestamp' column in the weight CSV")
    
    # Parse timestamps and set as index
    df_weight["Timestamp"] = pd.to_datetime(df_weight["Timestamp"])
    df_weight = df_weight.set_index("Timestamp").sort_index()
    
    # Convert all columns to numeric
    df_weight = df_weight.apply(pd.to_numeric, errors="coerce")
    
    # Resample to the requested frequency
    df_resampled = df_weight.resample(resample_freq, label="left", closed="left").mean()
    
    # Calculate days since experiment start
    days_since_start = (df_resampled.index - df_resampled.index.min()).total_seconds() / (24 * 3600)
    df_resampled["Days"] = days_since_start
    
    # Read randomization file and filter for this experiment
    randomization = pd.read_excel(randomization_path)
    experiment_name = f"CamelinaMAGIC_{experiment_num}.0"
    exp_randomization = randomization[randomization['Experiment'] == experiment_name].copy()
    
    # Create mappings of sample name to treatment and genotype
    treatment_map = dict(zip(exp_randomization['Name'], exp_randomization['Treatment']))
    genotype_map = dict(zip(exp_randomization['Name'], exp_randomization['Genotype']))
    
    # Transform from wide to long format
    # Reset index to have Days as a column
    df_resampled = df_resampled.reset_index(drop=True)
    
    # Melt the DataFrame to long format
    id_vars = ['Days']
    value_vars = [col for col in df_resampled.columns if col != 'Days']
    
    df_long = pd.melt(
        df_resampled,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='Sample_Column',
        value_name='Weight'
    )
    
    # Extract sample name from column name (e.g., "Camelina01 - Weight (g)" -> "Camelina01")
    df_long['Sample'] = df_long['Sample_Column'].str.extract(r'(Camelina\d+)')[0]
    
    # Add experiment number
    df_long['Experiment'] = experiment_num
    
    # Add treatment and genotype information
    df_long['Treatment'] = df_long['Sample'].map(treatment_map)
    df_long['Genotype'] = df_long['Sample'].map(genotype_map)
    
    # Drop the Sample_Column as we don't need it anymore
    df_long = df_long.drop(columns=['Sample_Column'])
    
    # Reorder columns for clarity
    df_long = df_long[['Days', 'Sample', 'Weight', 'Experiment', 'Treatment', 'Genotype']]
    
    # Remove rows with NaN weights
    df_long = df_long.dropna(subset=['Weight'])
    
    return df_long


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
    
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Filter for drought samples only
    df_drought = df[df['Treatment'] == 'DR_100'].copy()
    
    if df_drought.empty:
        raise ValueError("No drought samples found in the DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique genotypes and assign colors
    genotypes = sorted(df_drought['Genotype'].unique())
    # Use a predefined list of colors
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [color_list[i % len(color_list)] for i in range(len(genotypes))]
    
    # Plot each genotype
    for i, genotype in enumerate(genotypes):
        genotype_data = df_drought[df_drought['Genotype'] == genotype]
        
        # Calculate mean and std for each time point
        stats = genotype_data.groupby('Days')['Weight'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate 95% confidence interval
        # CI = mean ± 1.96 * (std / sqrt(n))
        stats['ci'] = 1.96 * stats['std'] / (stats['count'] ** 0.5)
        stats['lower'] = stats['mean'] - stats['ci']
        stats['upper'] = stats['mean'] + stats['ci']
        
        # Plot mean line
        ax.plot(stats['Days'], stats['mean'], label=genotype, color=colors[i], linewidth=2)
        
        # Plot confidence interval as shaded area
        ax.fill_between(stats['Days'], stats['lower'], stats['upper'], 
                        color=colors[i], alpha=0.2)
    
    ax.set_xlabel("Days since experiment start")
    ax.set_ylabel("Weight (g)")
    ax.legend(loc='best', title='Genotype')
    
    if title:
        ax.set_title(title)
    else:
        exp_num = df_drought['Experiment'].iloc[0]
        ax.set_title(f"Drought Samples by Genotype - Experiment {exp_num}")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    return fig, ax


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
    
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Filter for control samples only
    df_control = df[df['Treatment'] == 'Control'].copy()
    
    if df_control.empty:
        raise ValueError("No control samples found in the DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique genotypes and assign colors
    genotypes = sorted(df_control['Genotype'].unique())
    # Use a predefined list of colors
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [color_list[i % len(color_list)] for i in range(len(genotypes))]
    
    # Plot each genotype
    for i, genotype in enumerate(genotypes):
        genotype_data = df_control[df_control['Genotype'] == genotype]
        
        # Calculate mean and std for each time point
        stats = genotype_data.groupby('Days')['Weight'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate 95% confidence interval
        # CI = mean ± 1.96 * (std / sqrt(n))
        stats['ci'] = 1.96 * stats['std'] / (stats['count'] ** 0.5)
        stats['lower'] = stats['mean'] - stats['ci']
        stats['upper'] = stats['mean'] + stats['ci']
        
        # Plot mean line
        ax.plot(stats['Days'], stats['mean'], label=genotype, color=colors[i], linewidth=2)
        
        # Plot confidence interval as shaded area
        ax.fill_between(stats['Days'], stats['lower'], stats['upper'], 
                        color=colors[i], alpha=0.2)
    
    ax.set_xlabel("Days since experiment start")
    ax.set_ylabel("Weight (g)")
    ax.legend(loc='best', title='Genotype')
    
    if title:
        ax.set_title(title)
    else:
        exp_num = df_control['Experiment'].iloc[0]
        ax.set_title(f"Control Samples by Genotype - Experiment {exp_num}")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    return fig, ax


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
    
    # Validate paths
    if not os.path.exists(transpiration_csv_path):
        raise FileNotFoundError(f"Transpiration CSV not found: {transpiration_csv_path}")
    if not os.path.exists(randomization_path):
        raise FileNotFoundError(f"Randomization file not found: {randomization_path}")
    
    # Read the transpiration CSV
    df_transpiration = pd.read_csv(transpiration_csv_path)
    if "Timestamp" not in df_transpiration.columns:
        raise ValueError("Expected a 'Timestamp' column in the transpiration CSV")
    
    # Parse timestamps and set as index
    df_transpiration["Timestamp"] = pd.to_datetime(df_transpiration["Timestamp"])
    df_transpiration = df_transpiration.set_index("Timestamp").sort_index()
    
    # Convert all columns to numeric
    df_transpiration = df_transpiration.apply(pd.to_numeric, errors="coerce")
    
    # Calculate days since experiment start
    days_since_start = (df_transpiration.index - df_transpiration.index.min()).total_seconds() / (24 * 3600)
    df_transpiration["Days"] = days_since_start
    
    # Read randomization file and filter for this experiment
    randomization = pd.read_excel(randomization_path)
    experiment_name = f"CamelinaMAGIC_{experiment_num}.0"
    exp_randomization = randomization[randomization['Experiment'] == experiment_name].copy()
    
    # Create mappings of sample name to treatment and genotype
    treatment_map = dict(zip(exp_randomization['Name'], exp_randomization['Treatment']))
    genotype_map = dict(zip(exp_randomization['Name'], exp_randomization['Genotype']))
    
    # Transform from wide to long format
    # Reset index to have Days as a column
    df_transpiration = df_transpiration.reset_index(drop=True)
    
    # Melt the DataFrame to long format
    id_vars = ['Days']
    value_vars = [col for col in df_transpiration.columns if col != 'Days']
    
    df_long = pd.melt(
        df_transpiration,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='Sample_Column',
        value_name='Transpiration'
    )
    
    # Extract sample name from column name (e.g., "Camelina01 - Daily Transpiration (g)" -> "Camelina01")
    df_long['Sample'] = df_long['Sample_Column'].str.extract(r'(Camelina\d+)')[0]
    
    # Add experiment number
    df_long['Experiment'] = experiment_num
    
    # Add treatment and genotype information
    df_long['Treatment'] = df_long['Sample'].map(treatment_map)
    df_long['Genotype'] = df_long['Sample'].map(genotype_map)
    
    # Drop the Sample_Column as we don't need it anymore
    df_long = df_long.drop(columns=['Sample_Column'])
    
    # Reorder columns for clarity
    df_long = df_long[['Days', 'Sample', 'Transpiration', 'Experiment', 'Treatment', 'Genotype']]
    
    # Remove rows with NaN transpiration
    df_long = df_long.dropna(subset=['Transpiration'])
    
    return df_long


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
    
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Filter for drought samples only
    df_drought = df[df['Treatment'] == 'DR_100'].copy()
    
    if df_drought.empty:
        raise ValueError("No drought samples found in the DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique genotypes and assign colors
    genotypes = sorted(df_drought['Genotype'].unique())
    # Use a predefined list of colors
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [color_list[i % len(color_list)] for i in range(len(genotypes))]
    
    # Plot each genotype
    for i, genotype in enumerate(genotypes):
        genotype_data = df_drought[df_drought['Genotype'] == genotype]
        
        # Calculate mean and std for each time point
        stats = genotype_data.groupby('Days')['Transpiration'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate 95% confidence interval
        # CI = mean ± 1.96 * (std / sqrt(n))
        stats['ci'] = 1.96 * stats['std'] / (stats['count'] ** 0.5)
        stats['lower'] = stats['mean'] - stats['ci']
        stats['upper'] = stats['mean'] + stats['ci']
        
        # Plot mean line
        ax.plot(stats['Days'], stats['mean'], label=genotype, color=colors[i], linewidth=2)
        
        # Plot confidence interval as shaded area
        ax.fill_between(stats['Days'], stats['lower'], stats['upper'], 
                        color=colors[i], alpha=0.2)
    
    ax.set_xlabel("Days since experiment start")
    ax.set_ylabel("Daily Transpiration (g)")
    ax.legend(loc='best', title='Genotype')
    
    if title:
        ax.set_title(title)
    else:
        exp_num = df_drought['Experiment'].iloc[0]
        ax.set_title(f"Drought Samples - Daily Transpiration by Genotype - Experiment {exp_num}")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    return fig, ax


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
    
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Filter for control samples only
    df_control = df[df['Treatment'] == 'Control'].copy()
    
    if df_control.empty:
        raise ValueError("No control samples found in the DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique genotypes and assign colors
    genotypes = sorted(df_control['Genotype'].unique())
    # Use a predefined list of colors
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [color_list[i % len(color_list)] for i in range(len(genotypes))]
    
    # Plot each genotype
    for i, genotype in enumerate(genotypes):
        genotype_data = df_control[df_control['Genotype'] == genotype]
        
        # Calculate mean and std for each time point
        stats = genotype_data.groupby('Days')['Transpiration'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate 95% confidence interval
        # CI = mean ± 1.96 * (std / sqrt(n))
        stats['ci'] = 1.96 * stats['std'] / (stats['count'] ** 0.5)
        stats['lower'] = stats['mean'] - stats['ci']
        stats['upper'] = stats['mean'] + stats['ci']
        
        # Plot mean line
        ax.plot(stats['Days'], stats['mean'], label=genotype, color=colors[i], linewidth=2)
        
        # Plot confidence interval as shaded area
        ax.fill_between(stats['Days'], stats['lower'], stats['upper'], 
                        color=colors[i], alpha=0.2)
    
    ax.set_xlabel("Days since experiment start")
    ax.set_ylabel("Daily Transpiration (g)")
    ax.legend(loc='best', title='Genotype')
    
    if title:
        ax.set_title(title)
    else:
        exp_num = df_control['Experiment'].iloc[0]
        ax.set_title(f"Control Samples - Daily Transpiration by Genotype - Experiment {exp_num}")
    
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
    
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Filter for the specific genotype
    df_genotype = df[df['Genotype'] == genotype].copy()
    
    if df_genotype.empty:
        raise ValueError(f"No data found for genotype '{genotype}' in the DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique treatments and assign colors
    treatments = sorted(df_genotype['Treatment'].unique())
    treatment_colors = {'Control': '#2ca02c', 'DR_100': '#d62728'}  # Green for control, red for drought
    
    # Plot each treatment
    for treatment in treatments:
        treatment_data = df_genotype[df_genotype['Treatment'] == treatment]
        
        # Calculate mean and std for each time point
        stats = treatment_data.groupby('Days')['Weight'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate 95% confidence interval
        # CI = mean ± 1.96 * (std / sqrt(n))
        stats['ci'] = 1.96 * stats['std'] / (stats['count'] ** 0.5)
        stats['lower'] = stats['mean'] - stats['ci']
        stats['upper'] = stats['mean'] + stats['ci']
        
        # Get color for this treatment
        color = treatment_colors.get(treatment, '#1f77b4')
        
        # Plot mean line
        label = 'Control' if treatment == 'Control' else 'Drought'
        ax.plot(stats['Days'], stats['mean'], label=label, color=color, linewidth=2)
        
        # Plot confidence interval as shaded area
        ax.fill_between(stats['Days'], stats['lower'], stats['upper'], 
                        color=color, alpha=0.2)
    
    ax.set_xlabel("Days since experiment start")
    ax.set_ylabel("Weight (g)")
    ax.legend(loc='best', title='Treatment')
    
    if title:
        ax.set_title(title)
    else:
        exp_num = df_genotype['Experiment'].iloc[0]
        ax.set_title(f"Weight for {genotype} - Control vs Drought - Experiment {exp_num}")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    return fig, ax


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
    
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Filter for the specific genotype
    df_genotype = df[df['Genotype'] == genotype].copy()
    
    if df_genotype.empty:
        raise ValueError(f"No data found for genotype '{genotype}' in the DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique treatments and assign colors
    treatments = sorted(df_genotype['Treatment'].unique())
    treatment_colors = {'Control': '#2ca02c', 'DR_100': '#d62728'}  # Green for control, red for drought
    
    # Plot each treatment
    for treatment in treatments:
        treatment_data = df_genotype[df_genotype['Treatment'] == treatment]
        
        # Calculate mean and std for each time point
        stats = treatment_data.groupby('Days')['Transpiration'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate 95% confidence interval
        # CI = mean ± 1.96 * (std / sqrt(n))
        stats['ci'] = 1.96 * stats['std'] / (stats['count'] ** 0.5)
        stats['lower'] = stats['mean'] - stats['ci']
        stats['upper'] = stats['mean'] + stats['ci']
        
        # Get color for this treatment
        color = treatment_colors.get(treatment, '#1f77b4')
        
        # Plot mean line
        label = 'Control' if treatment == 'Control' else 'Drought'
        ax.plot(stats['Days'], stats['mean'], label=label, color=color, linewidth=2)
        
        # Plot confidence interval as shaded area
        ax.fill_between(stats['Days'], stats['lower'], stats['upper'], 
                        color=color, alpha=0.2)
    
    ax.set_xlabel("Days since experiment start")
    ax.set_ylabel("Daily Transpiration (g)")
    ax.legend(loc='best', title='Treatment')
    
    if title:
        ax.set_title(title)
    else:
        exp_num = df_genotype['Experiment'].iloc[0]
        ax.set_title(f"Daily Transpiration for {genotype} - Control vs Drought - Experiment {exp_num}")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    return fig, ax


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
    
    # Validate path
    if not os.path.exists(weather_csv_path):
        raise FileNotFoundError(f"Weather station CSV not found: {weather_csv_path}")
    
    # Read the weather CSV
    df_weather = pd.read_csv(weather_csv_path)
    if "Timestamp" not in df_weather.columns:
        raise ValueError("Expected a 'Timestamp' column in the weather CSV")
    
    # Parse timestamps and set as index
    df_weather["Timestamp"] = pd.to_datetime(df_weather["Timestamp"])
    df_weather = df_weather.set_index("Timestamp").sort_index()
    
    # Rename columns to simpler names (remove the full prefix)
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
    
    # Convert all columns to numeric
    df_weather = df_weather.apply(pd.to_numeric, errors="coerce")
    
    # Resample to the requested frequency (take mean of each interval)
    df_resampled = df_weather.resample(resample_freq, label="left", closed="left").mean()
    
    # Calculate days since experiment start
    days_since_start = (df_resampled.index - df_resampled.index.min()).total_seconds() / (24 * 3600)
    df_resampled["Days"] = days_since_start
    
    # Add experiment number
    df_resampled['Experiment'] = experiment_num
    
    # Reset index to have Days as a regular column
    df_resampled = df_resampled.reset_index(drop=True)
    
    # Reorder columns for clarity
    df_resampled = df_resampled[['Days', 'PARLight', 'RH', 'Temp', 'VPD', 'Experiment']]
    
    # Remove rows with all NaN values
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
    
    # Plot PAR Light
    axes[0].plot(df['Days'], df['PARLight'], color='#ff7f0e', linewidth=1)
    axes[0].set_ylabel('PAR Light\n(µmol/m²/s)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Relative Humidity
    axes[1].plot(df['Days'], df['RH'], color='#2ca02c', linewidth=1)
    axes[1].set_ylabel('Relative Humidity\n(%)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot Temperature
    axes[2].plot(df['Days'], df['Temp'], color='#d62728', linewidth=1)
    axes[2].set_ylabel('Temperature\n(°C)')
    axes[2].grid(True, alpha=0.3)
    
    # Plot VPD
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
