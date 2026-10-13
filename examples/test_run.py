import sys

sys.path.append('src')

import pyspc
import matplotlib.pyplot as plt


file_path = '/home/galena/vscode/plastic_adhesive_model_calibration/raw_data/Aggregated/20260821/TN_20260821_agg_001.nsa'

df_l, df_u = pyspc.preprocess_force_data_2575_method(file_path)

# pyspc.plot_test(df_l, df_u)

particle_properties = pyspc.generate_particle_properties(radius=0.25e-6)  # Example radius in meters

(E_fit, lp_fit), model_forces = pyspc.run_optimization(df_l, df_u, particle_properties=particle_properties)

print("\nOptimization Complete!")
print(f"Optimized Parameters: E* = {E_fit:.2e}, lam_p = {lp_fit:.4f}")

pyspc.plot_eepa_fit(df_l, df_u, model_forces, (E_fit, lp_fit))