import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .utilities import change_raw_data_units


def read_data(input_nsa_file):
    data_lines = []
    found_data_section = False
    
    with open(input_nsa_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if '(nN)' in line:
                found_data_section = True
                continue
            
            if found_data_section:
                clean_line = line.strip()
                if clean_line:
                    data_lines.append(clean_line)

    df = pd.read_csv(
        io.StringIO('\n'.join(data_lines)), 
        header=None,
        names=['id', 'indentation_um', 'force_nN', 'extra']
    )

    df = df[['id', 'indentation_um', 'force_nN']]
    return df


def remove_zero_force(df, force_col='force_nN', mode='start', window_size_percentage=0.03, gradient_threshold_percentage=0.015):
    """
    Removes zero-force non-contact regions from data curves.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    force_col : str
        Name of the force column.
    mode : str
        'start' - Trims initial pre-contact data (loading curve).
        'end'   - Trims trailing post-contact noise tail (unloading curve).
    window_size_percentage : float
        Window size as a fraction of total sample count.
    gradient_threshold_percentage : float
        Gradient threshold as a fraction of max force.
    """
    if df.empty or force_col not in df.columns:
        return df

    gradient_threshold = df[force_col].max() * gradient_threshold_percentage

    # Extract the force column as a 1D numpy array
    force_values = df[force_col].to_numpy()
    n_samples = len(force_values)

    # Ensure window_size is at least 2 for linear regression
    window_size = int(n_samples * window_size_percentage)
    if window_size < 2:
        raise ValueError(f"Window size ({window_size}) must be at least 2. Increase window_size_percentage or sample count.")
    x_window = np.arange(window_size)

    if mode == 'start':
        # Scan forward from the start to find initial contact
        for i in range(n_samples - window_size + 1):
            y_window = force_values[i : i + window_size]
            gradient, _ = np.polyfit(x_window, y_window, 1)
            
            if abs(gradient) > gradient_threshold:
                return df.iloc[i:].reset_index(drop=True)

    elif mode == 'end':
        # Scan backward from the end to find where contact was lost
        for i in range(n_samples - window_size, -1, -1):
            y_window = force_values[i : i + window_size]
            gradient, _ = np.polyfit(x_window, y_window, 1)
            
            if abs(gradient) > gradient_threshold:
                # Keep up to the end of the last active contact window
                return df.iloc[: i + window_size].reset_index(drop=True)

    return df


def _find_unloading_start(df_post, force_col, window_size, threshold_frac):
    """
    Calculates the index where the unloading phase begins based on the force gradient.
    Returns the index if found, otherwise returns None.
    """
    # Calculate smoothed gradient (dP/dt)
    force_diff = df_post[force_col].diff().fillna(0)
    smoothed_grad = force_diff.rolling(window=window_size, center=True, min_periods=1).mean()
    
    # Define threshold for unloading
    min_grad = smoothed_grad.min()
    unload_threshold = min_grad * threshold_frac
    
    # Find indices steeper than the threshold
    steep_indices = smoothed_grad[smoothed_grad <= unload_threshold].index
    
    return steep_indices[0] if not steep_indices.empty else None


def split_indentation_phases(df, force_col='force_nN', dist_col='indentation_um', 
                             window_size=3, gradient_threshold_frac=0.55):
    """
    Splits indentation data into loading, backlash, and unloading sections.
    """
    empty_df = pd.DataFrame(columns=df.columns)
    
    if df.empty:
        return empty_df.copy(), empty_df.copy(), empty_df.copy()
        
    # 1. Identify Loading Phase (Ends at max displacement)
    idx_max_h = df[dist_col].idxmax()
    df_loading = df.loc[:idx_max_h].copy()
    
    # 2. Isolate post-loading data
    df_post = df.loc[idx_max_h + 1:].copy()
    if df_post.empty:
        return df_loading, empty_df.copy(), empty_df.copy()
        
    # 3. Calculate where unloading begins
    idx_unload_start = _find_unloading_start(
        df_post, force_col, window_size, gradient_threshold_frac
    )
    
    # 4. Split into backlash and unloading
    if idx_unload_start is not None:
        df_backlash = df.loc[idx_max_h + 1 : idx_unload_start - 1].copy()
        df_unloading = df.loc[idx_unload_start:].copy()
    else:
        # Fallback if no steep gradient is detected
        df_backlash = df_post.copy()
        df_unloading = empty_df.copy()
        
    return df_loading, df_backlash, df_unloading


def offset_unloading_by_backlash(df_b, df_u, dist_col):
    """
    Offsets the unloading curve by the maximum indentation of the backlash curve.
    """
    if df_b.empty or df_u.empty:
        return df_u

    max_backlash_indent = df_b[dist_col].max()
    max_unloading_indent = df_u[dist_col].max()
    offset = max_backlash_indent - max_unloading_indent
    df_u[dist_col] += offset
    return df_u


def zero_indentation(df_l, df_u, df_b=None, dist_col='indentation_m'):
    """
    Adjusts the indentation values so that the minimum indentation is zero.
    """
    if df_l.empty or dist_col not in df_l.columns:
        return df_l
    if df_u.empty or dist_col not in df_u.columns:
        return df_u
    if df_b is not None and (df_b.empty or dist_col not in df_b.columns):
        return df_b

    min_indent = min(df_l[dist_col].min(), df_u[dist_col].min())
    if df_b is not None:
        min_indent = min(min_indent, df_b[dist_col].min())

    df_l[dist_col] -= min_indent
    df_u[dist_col] -= min_indent
    if df_b is not None:
        df_b[dist_col] -= min_indent

    return df_l, df_u, df_b


def preprocess_force_data(file_path, threshold = 0.07, force_col='force_nN', dist_col='indentation_um', window_size_percentage=0.04, gradient_threshold_percentage=0.015):
    """
    Preprocesses indentation-force data from a given file path.
    Steps:
    1. Reads the raw data from the file.
    2. Splits the data into loading, backlash, and unloading phases.
    3. Removes zero-force regions from loading and unloading curves.
    4. Offsets the unloading curve to align with the loading curve.
    5. Zeros the indentation values for both loading and unloading curves.
    Returns:
    --------
    df_l : pd.DataFrame
        Preprocessed loading curve data.
    df_u : pd.DataFrame
        Preprocessed unloading curve data.
    """
    # First remove the pre-contact (lead-up) data
    data = read_data(file_path)

    # Identify and split into loading, backlash, and unloading phases
    df_l, df_b, df_u = split_indentation_phases(data)

    # Remove zero-force regions from loading and unloading curves
    df_l = remove_zero_force(df_l, force_col=force_col, mode='start', 
                             window_size_percentage=window_size_percentage, 
                             gradient_threshold_percentage=gradient_threshold_percentage)
    
    df_u = remove_zero_force(df_u, force_col=force_col, mode='end', 
                             window_size_percentage=window_size_percentage, 
                             gradient_threshold_percentage=gradient_threshold_percentage)

    # Move unloading curve to align with loading curve
    df_u = offset_unloading_by_backlash(df_b, df_u, dist_col=dist_col)

    # Zero the indentation values for both loading and unloading curves
    df_l, df_u, _ = zero_indentation(df_l, df_u, dist_col=dist_col)

    return df_l, df_u


def preprocess_force_data_2575_method(file_path, threshold = 0.05, force_col='force_nN', dist_col='indentation_um', window_size_percentage=0.04, gradient_threshold_percentage=0.015):
    """
    Preprocesses indentation-force data from a given file path.
    Steps:
    1. Reads the raw data from the file.
    2. Splits the data into loading, backlash, and unloading phases.
    3. Removes zero-force regions from loading and unloading curves.
    4. Offsets the unloading curve to align with the loading curve.
    5. Zeros the indentation values for both loading and unloading curves.
    Returns:
    --------
    df_l : pd.DataFrame
        Preprocessed loading curve data.
    df_u : pd.DataFrame
        Preprocessed unloading curve data.
    """
    # First remove the pre-contact (lead-up) data
    data = read_data(file_path)

    data_in_si_units = change_raw_data_units(data)

    si_force = 'force_N'
    si_dist = 'indentation_m'

    cleaner_data = remove_zero_force_2575_method(data_in_si_units, force_col=si_force, dist_col=si_dist)

    df_l_corrected = apply_compliance_correction(df_l, df_baseline)
    df_u_corrected = apply_compliance_correction(df_u, df_baseline)

    df_l, df_b, df_u = split_indentation_phases(cleaner_data, force_col=si_force, dist_col=si_dist)

    df_u = offset_unloading_by_backlash(df_b, df_u, dist_col=si_dist)
    df_l, df_u, _ = zero_indentation(df_l, df_u, dist_col=si_dist)

    return df_l, df_u


def preprocess_paired_data(sample_file_path, baseline_file_path, force_col='force_N', dist_col='indentation_m'):
    """
    Preprocesses a particle indentation file by pairing it with its full-cycle 
    baseline (loading, backlash, and unloading).
    """
    # 1. Read and convert BOTH to SI units
    df_samp = change_raw_data_units(read_data(sample_file_path))
    df_base = change_raw_data_units(read_data(baseline_file_path))

    # 2. Clean the air-phase and zero the sensor drift for BOTH
    clean_samp = remove_zero_force_2575_method(df_samp, force_col=force_col, dist_col=dist_col)
    clean_base = remove_zero_force_2575_method(df_base, force_col=force_col, dist_col=dist_col)

    # 3. Split BOTH curves into their three physical phases
    samp_l, samp_b, samp_u = split_indentation_phases(clean_samp, force_col=force_col, dist_col=dist_col)
    base_l, base_b, base_u = split_indentation_phases(clean_base, force_col=force_col, dist_col=dist_col)

    # 4. Phase-by-Phase Compliance Correction
    # Maps Loading to Loading, Backlash to Backlash, and Unloading to Unloading
    samp_l_corr = apply_compliance_correction(samp_l, base_l, force_col=force_col, dist_col=dist_col)
    samp_b_corr = apply_compliance_correction(samp_b, base_b, force_col=force_col, dist_col=dist_col)
    samp_u_corr = apply_compliance_correction(samp_u, base_u, force_col=force_col, dist_col=dist_col)

    # 5. Apply Backlash Offset & Zeroing using the CORRECTED arrays
    # This mathematically removes machine hysteresis from the final particle curve
    samp_u_final = offset_unloading_by_backlash(samp_b_corr, samp_u_corr, dist_col=dist_col)
    samp_l_final, samp_u_final, _ = zero_indentation(samp_l_corr, samp_u_final, samp_b_corr, dist_col=dist_col)

    # 6. Final safety filter to prevent any < 0.0 values causing optimizer crashes
    samp_l_final = samp_l_final[samp_l_final[dist_col] >= 0.0]
    samp_u_final = samp_u_final[samp_u_final[dist_col] >= 0.0]

    return samp_l_final, samp_u_final


def remove_zero_force_2575_method(data, force_col, dist_col):

    # First we find the max indentation point to get max force
    idx_max_h = data[dist_col].idxmax()
    force_max = data.loc[idx_max_h, force_col]

    # From maximum force we find the 25% and 75% force points
    force_25 = 0.25 * force_max
    force_75 = 0.75 * force_max

    # Split the loading and unloading primitively by the max indentation point
    df_l = data.loc[:idx_max_h].copy() 
    df_u = data.loc[idx_max_h:].copy()

    # Find the indices where the force crosses these thresholds for both loading and unloading
    idx_25_l = df_l[force_col].sub(force_25).abs().idxmin()
    idx_75_l = df_l[force_col].sub(force_75).abs().idxmin()
    idx_25_u = df_u[force_col].sub(force_25).abs().idxmin()
    idx_75_u = df_u[force_col].sub(force_75).abs().idxmin()

    # Fit to loading curve
    df_l_fit = df_l.loc[idx_25_l:idx_75_l]
    coeffs_l = np.polyfit(df_l_fit[dist_col], df_l_fit[force_col], 1)
    m_l = coeffs_l[0]
    k_l = coeffs_l[1]

    # Fit to unloading curve
    df_u_fit = df_u.loc[idx_75_u:idx_25_u]
    coeffs_u = np.polyfit(df_u_fit[dist_col], df_u_fit[force_col], 1)
    m_u = coeffs_u[0]
    k_u = coeffs_u[1]

    # Now find the x axis intercept for both loading and unloading curves
    h0_l = -k_l / m_l
    h0_u = -k_u / m_u

    # Remove data points with indentations smaller than the intercept value
    df_l_clean = df_l[df_l[dist_col] >= h0_l]
    df_u_clean = df_u[df_u[dist_col] >= h0_u]

    return pd.concat([df_l_clean, df_u_clean], axis=0)
    

def apply_compliance_correction_loading_only(df_sample, df_baseline, force_col='force_N', dist_col='indentation_m'):
    """
    Subtracts the substrate (Teflon) deflection from the sample data.
    Both dataframes must already be preprocessed and in SI units.
    """
    df_corrected = df_sample.copy()
    
    # 1. Isolate the loading portion of the baseline to create a compliance map
    idx_max_base = df_baseline[dist_col].idxmax()
    base_load = df_baseline.loc[:idx_max_base].copy()
    
    # Sort values to ensure the interpolation function works smoothly
    base_load = base_load.sort_values(by=force_col)
    
    # 2. Create an interpolation function: Force -> Substrate Displacement
    # 'extrapolate' handles cases where the sample force slightly exceeds the baseline force
    compliance_func = interp1d(
        base_load[force_col], 
        base_load[dist_col], 
        kind='linear', 
        fill_value='extrapolate'
    )
    
    # 3. Calculate how much the Teflon deflected at these specific force values
    substrate_deflection = compliance_func(df_corrected[force_col])
    
    # Prevent subtracting negative deflections if sensor noise dips below 0
    substrate_deflection = np.maximum(0, substrate_deflection)
    
    # 4. Subtract substrate displacement from the sample displacement
    df_corrected[dist_col] = df_corrected[dist_col] - substrate_deflection
    
    # Clean up any negative displacements caused by the subtraction at the very beginning
    df_corrected = df_corrected[df_corrected[dist_col] >= 0.0]
    
    return df_corrected


def apply_compliance_correction(df_sample_phase, df_base_phase, force_col='force_N', dist_col='indentation_m'):
    """
    Subtracts the machine/substrate deflection from a specific isolated phase 
    (loading, backlash, or unloading).
    """
    df_corrected = df_sample_phase.copy()
    
    if df_corrected.empty or df_base_phase.empty:
        return df_corrected
        
    # Sort baseline by force to ensure monotonically increasing X-values for interpolation
    df_base_phase = df_base_phase.sort_values(by=force_col).drop_duplicates(subset=[force_col])
    
    # Create interpolation map: Force -> Machine Displacement
    compliance_func = interp1d(
        df_base_phase[force_col], 
        df_base_phase[dist_col], 
        kind='linear', 
        fill_value='extrapolate'
    )
    
    # Calculate how much the machine deflected at the sample's specific force values
    substrate_deflection = compliance_func(df_corrected[force_col])
    
    # Subtract machine displacement from the sample displacement
    df_corrected[dist_col] = df_corrected[dist_col] - np.maximum(0, substrate_deflection)
    
    return df_corrected


def preprocess_baseline_data(file_path, force_col='force_N', dist_col='indentation_m'):
    """
    Dedicated preprocessor for loading-only baseline (Teflon) files.
    """
    data = read_data(file_path)
    data = change_raw_data_units(data)

    # 1. Baseline drift correction (zero the sensor)
    n_baseline_points = max(5, int(len(data) * 0.05))
    baseline_offset = data[force_col].head(n_baseline_points).mean()
    data[force_col] = data[force_col] - baseline_offset

    # 2. Find maximum force and calculate 25/75 thresholds
    force_max = data[force_col].max()
    force_min = data[force_col].min()

    amp = force_max - force_min
    force_25 = force_min + (0.25 * amp)
    force_75 = force_min + (0.75 * amp)

    # 3. Find indices closest to 25% and 75%
    idx_25 = data[force_col].sub(force_25).abs().idxmin()
    idx_75 = data[force_col].sub(force_75).abs().idxmin()

    # --- FIXED: Ensure indices are in the correct order to prevent empty slices! ---
    idx_start = min(idx_25, idx_75)
    idx_end = max(idx_25, idx_75)
    
    df_fit = data.loc[idx_start:idx_end]

    # 4. Fit linear curve to find true contact point (h0) safely
    if len(df_fit) >= 2:
        coeffs = np.polyfit(df_fit[dist_col], df_fit[force_col], 1)
        h0 = -coeffs[1] / coeffs[0] if coeffs[0] != 0 else df_fit[dist_col].iloc[0]
    else:
        # Fallback if the data is too sparse to fit a line
        h0 = data.loc[idx_start, dist_col]

    # 5. Clean pre-contact data and shift x-axis so contact is at exactly 0.0 m
    df_clean = data[data[dist_col] >= h0].copy()
    if not df_clean.empty:
        df_clean[dist_col] = df_clean[dist_col] - h0

    return df_clean