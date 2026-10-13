import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt


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


def remove_zero_force(df, force_col='force_nN', mode='start', window_size_percentage=0.02, gradient_threshold_percentage=0.015):
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
                             window_size=3, gradient_threshold_frac=0.5):
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


def offset_unloading_by_backlash(df_b, df_u, dist_col='indentation_um'):
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


def zero_indentation(df_l, df_u, df_b=None, dist_col='indentation_um'):
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


def preprocess_force_data(file_path, threshold = 0.05, force_col='force_nN', dist_col='indentation_um', window_size_percentage=0.04, gradient_threshold_percentage=0.015):
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

    cleaner_data = remove_zero_force_2575_method(data)

    df_l, df_b, df_u = split_indentation_phases(cleaner_data)

    # Move unloading curve to align with loading curve
    df_u = offset_unloading_by_backlash(df_b, df_u, dist_col=dist_col)

    # Zero the indentation values for both loading and unloading curves
    df_l, df_u, _ = zero_indentation(df_l, df_u, dist_col=dist_col)

    return df_l, df_u


def remove_zero_force_2575_method(data, force_col='force_nN', dist_col='indentation_um'):

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
    