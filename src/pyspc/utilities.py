from __future__ import annotations
from pathlib import Path
import pandas as pd
import re
import os
import sys
from collections import defaultdict


def find_nsa_files_in_directory(directory: str, blacklisted_strs: list[str] = None) -> list[str]:
    """Recursively finds .nsa files, excluding any containing blacklisted strings."""
    
    # Safely set the default blacklist
    if blacklisted_strs is None:
        blacklisted_strs = ['baseline']
        
    return [
        str(path) 
        for path in Path(directory).rglob('*.nsa') 
        # Keep the file only if ALL blacklisted strings are NOT in the filename
        if all(bad_str.lower() not in path.name.lower() for bad_str in blacklisted_strs)
    ]


def get_particle_radius_from_dir(nsa_file_path: str) -> float:
    """
    Finds the corresponding diameter.xlsx/csv file in the same directory, 
    matches the data_number from the .nsa filename, and returns the radius in meters.
    Raises an error if the file, column, or data number is missing.
    """
    nsa_path = Path(nsa_file_path)
    parent_dir = nsa_path.parent
    filename = nsa_path.name
    
    # 1. Extract data_number from the .nsa filename (e.g., ..._006.nsa -> 6)
    match = re.search(r'_(\d+)(?:_[A-Za-z]+)?\.nsa$', filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract a numeric data_number from filename: {filename}")
    
    data_number = int(match.group(1))
    
    # 2. Find the diameter file
    d_files = list(parent_dir.glob('*diameter.xlsx')) + list(parent_dir.glob('*diameter.csv'))
    if not d_files:
        raise FileNotFoundError(f"No '*diameter.xlsx' or '*diameter.csv' found in directory: {parent_dir}")
        
    d_file = d_files[0]
    
    # 3. Load the data
    if d_file.suffix.lower() == '.csv':
        df = pd.read_csv(d_file)
    else:
        df = pd.read_excel(d_file)
        
    df.columns = df.columns.str.strip()
    
    # If the headers are messed up because of that top row, grab the next row down
    if 'data_number' not in df.columns:
        # Re-load skipping the first row
        if d_file.suffix.lower() == '.csv':
            df = pd.read_csv(d_file, header=1)
        else:
            df = pd.read_excel(d_file, header=1)
            
        df.columns = df.columns.str.strip()
    
        # Force the column to be numeric (turns the Japanese text row into NaN)
        if 'data_number' in df.columns:
            df['data_number'] = pd.to_numeric(df['data_number'], errors='coerce')
    
    # 4. Verify columns exist
    if 'data_number' not in df.columns or 'diameter' not in df.columns:
        raise KeyError(f"The file {d_file.name} is missing 'data_number' or 'diameter' columns.")
        
    # 5. Find the exact row matching the data_number
    row = df[df['data_number'] == data_number]
    if row.empty:
        raise ValueError(f"data_number {data_number} (from {filename}) was not found in {d_file.name}")
        
    diameter_um = row['diameter'].iloc[0]
    
    # 6. Convert diameter (um) -> radius (meters)
    radius_m = (diameter_um / 2.0) * 1e-6
    return radius_m


def build_preprocessing_registry(target_dir: str, base_dir: str, blacklisted_strs: list = None) -> pd.DataFrame:
    """
    Scans for NSA files, extracts their particle diameters, builds a DataFrame 
    manifest, prints a compact tree report, and waits for user confirmation.
    """
    if blacklisted_strs is None:
        blacklisted_strs = ['baseline']
        
    print("Scanning directories and extracting Excel data...")
    all_files = find_nsa_files_in_directory(target_dir, blacklisted_strs)

    registry = []
    
    for file_path in all_files:
        display_name = os.path.relpath(file_path, start=base_dir)
        
        # Get the radius using our strict utility function
        try:
            radius_m = get_particle_radius_from_dir(file_path)
            registry.append({
                'file_path': file_path,
                'display_name': display_name,
                'radius_m': radius_m,
            })
        except Exception as e:
            print(f"Skipping {display_name} -> {e}")
        
    df_registry = pd.DataFrame(registry)
    
    if df_registry.empty:
        sys.exit("No valid files found. Exiting.")

    # ==========================================
    # 1. Build and Print the Tree Report
    # ==========================================
    tree = defaultdict(list)
    for _, row in df_registry.iterrows():
        # Group by the folder path
        folder = os.path.dirname(row['display_name'])
        filename = os.path.basename(row['display_name'])
        tree[folder].append((filename, row['radius_m']))
        
    print("\n" + "="*50)
    print(" Data Pre-processing Summary")
    print("="*50)
    
    for folder, files in sorted(tree.items()):
        print(f"📁 {folder}/")
        
        # Compact view: Show max 3 files per folder if there are many
        if len(files) > 4:
            for fname, rad in files[:3]:
                print(f" ├── 📄 {fname} (D: {rad * 2 * 1e6:.2f} µm)")
            print(f" └── ... plus {len(files) - 3} more files")
        else:
            # Print all files if it's a short list
            for i, (fname, rad) in enumerate(files):
                connector = " └──" if i == len(files) - 1 else " ├──"
                print(f"{connector} 📄 {fname} (D: {rad * 2 * 1e6:.2f} µm)")
                
    # ==========================================
    # 2. Print Summary Statistics
    # ==========================================
    mean_d = df_registry['radius_m'].mean() * 2 * 1e6
    
    print("-" * 50)
    print(f"Total Folders: {len(tree)}")
    print(f"Total Files:   {len(df_registry)}")
    print(f"Average Diam:  {mean_d:.2f} µm")
    print("=" * 50)
    
    # ==========================================
    # 3. User Confirmation Prompt
    # ==========================================
    choice = input("\nDoes this look correct? Start processing? [y/N]: ").strip().lower()
    
    if choice != 'y':
        sys.exit("Processing aborted by user.")
        
    print("\nStarting optimization pipeline...\n")
    return df_registry


def generate_particle_properties(radius, youngs_modulus_bounds=(1e5, 200e9), plasticity_ratio_bounds=(0.01, 0.95)):
    """Generates a dictionary of particle properties for optimization."""
    return {
        'radius': radius,
        'poisson_ratio': 0.4,
        'youngs_modulus_bounds': youngs_modulus_bounds,
        'youngs_modulus': None,
        'plasticity_ratio_bounds': plasticity_ratio_bounds,
        'plasticity_ratio': None,
    }

def generate_optimization_settings(popsize=25, tol=0.001, workers=-1):
    """Generates a dictionary of optimization settings for differential evolution."""
    return {
        'popsize': popsize,
        'tol': tol,
        'workers': workers
    }


def generate_needle_properties():
    return {
        'youngs_modulus': 400e9,  # Young's modulus for tungsten (Pa)
        'poisson_ratio': 0.4,  # Poisson's ratio for tungsten
    }


def generate_substrate_properties():
    return {
        'youngs_modulus': 2000e9,  # Young's modulus for teflon (Pa)
        'poisson_ratio': 0.4,  # Poisson's ratio for teflon
    }

def calculate_effective_modulus(E1, nu1, E2, nu2):
    """Calculates the effective modulus for two materials in contact."""
    return 1 / ((1 - nu1**2) / E1 + (1 - nu2**2) / E2)


def calculate_r_star(radius, radius2=None):
    """Calculates the effective radius for two spheres in contact."""
    if radius2 is None:
        return radius  # For a single sphere, R* = R
    else:
        return (radius * radius2) / (radius + radius2)  # For two spheres, R* = (R1 * R2) / (R1 + R2)


def calculate_e_star(E1, nu1, E2, nu2):
    """Calculates the effective modulus for two materials in contact."""
    return 1 / ((1 - nu1**2) / E1 + (1 - nu2**2) / E2)


def change_raw_data_units(data, force_col_in='force_nN', dist_col_in='indentation_um'):
    """
    Converts raw experimental data into standard SI units (Meters and Newtons).
    Returns a copy of the DataFrame with new standardized columns.
    """
    df = data.copy()
    
    # Convert micrometers to meters (1 um = 1e-6 m)
    if dist_col_in in df.columns:
        df['indentation_m'] = df[dist_col_in] * 1e-6
        
    # Convert nanoNewtons to Newtons (1 nN = 1e-9 N)
    if force_col_in in df.columns:
        df['force_N'] = df[force_col_in] * 1e-9
        
    return df


def get_baseline_for_sample(sample_file_path):
    """
    Finds the baseline NSA file located in the same directory as the given sample file.
    """
    parent_dir = os.path.dirname(sample_file_path)
    
    # Search for any file containing 'baseline' in the filename
    baseline_files = [f for f in os.listdir(parent_dir) 
                      if 'baseline' in f.lower() and f.endswith('.nsa')]
                      
    if not baseline_files:
        raise FileNotFoundError(f"No baseline NSA file found in folder: {parent_dir}")
        
    return os.path.join(parent_dir, baseline_files[0])
