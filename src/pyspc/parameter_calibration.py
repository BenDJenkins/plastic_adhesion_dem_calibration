import os
import pandas as pd
from .utilities import generate_particle_properties, build_preprocessing_registry
from .force_data_preprocessing import preprocess_force_data, preprocess_force_data_2575_method
from .force_balance_indentor import run_optimization
from .plotting import create_base64_plot


def process_and_save_data(target_dir, base_dir, output_csv="optimized_results.csv"):
    """
    Finds NSA files, runs EEPA optimization, generates plots, 
    and saves all results to a CSV.
    """
    df_registry = build_preprocessing_registry(target_dir, base_dir, blacklisted_strs=['baseline'])

    results = []

    for index, row in df_registry.iterrows():
        file_path = row['file_path']
        display_name = row['display_name']
        radius_m = row['radius_m']
        
        print(f"\n[{index+1}/{len(df_registry)}] Processing: {display_name}")
        
        display_name = os.path.relpath(file_path, start=base_dir)
        print(f"\nProcessing: {display_name}")
        
        # 1. Preprocess and Optimize
        df_l, df_u = preprocess_force_data_2575_method(file_path)
        particle_properties = generate_particle_properties(radius=radius_m)

        (E_fit, lp_fit), model_forces = run_optimization(df_l, df_u, particle_properties=particle_properties)
        print(f"Optimized Parameters: E* = {E_fit:.2e}, lam_p = {lp_fit:.4f}")

        # 2. Generate base64 image
        b64_img = create_base64_plot(df_l, df_u, model_forces, (E_fit, lp_fit))
        
        # 3. Store result dictionary
        results.append({
            'file_path': file_path,
            'display_name': display_name,
            'E_fit': E_fit,
            'lp_fit': lp_fit,
            'b64_img': b64_img
        })

    # Save to DataFrame and CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved raw data and images to {output_csv}")
    return