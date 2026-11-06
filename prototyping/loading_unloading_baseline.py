import sys
import os

# Point Python to the 'src' directory exactly one level up from 'prototyping'
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.insert(0, src_path)

import matplotlib.pyplot as plt
import scienceplots
from pyspc import read_data, change_raw_data_units
from pyspc.force_data_preprocessing import remove_zero_force_2575_method, split_indentation_phases, apply_compliance_correction

# --- Settings and File Paths ---
sample_file_path = "raw_data/new/Monolithic/20261105/TN_20261105_mono_001.nsa" # Update with your file
baseline_file_path = "raw_data/new/Monolithic/20261105/TN_20261105_baseline_001.nsa" # Update with your file

force_col = 'force_N'
dist_col = 'indentation_m'

# --- 1. Data Loading & Cleaning ---
df_samp = change_raw_data_units(read_data(sample_file_path))
df_base = change_raw_data_units(read_data(baseline_file_path))

clean_samp = remove_zero_force_2575_method(df_samp, force_col=force_col, dist_col=dist_col)
clean_base = remove_zero_force_2575_method(df_base, force_col=force_col, dist_col=dist_col)

# --- 2. Phase Splitting ---
samp_l, samp_b, samp_u = split_indentation_phases(clean_samp, force_col=force_col, dist_col=dist_col)
base_l, base_b, base_u = split_indentation_phases(clean_base, force_col=force_col, dist_col=dist_col)

samp_l_corr = apply_compliance_correction(samp_l, base_l, force_col=force_col, dist_col=dist_col)
samp_b_corr = apply_compliance_correction(samp_b, base_b, force_col=force_col, dist_col=dist_col)
samp_u_corr = apply_compliance_correction(samp_u, base_u, force_col=force_col, dist_col=dist_col)

# --- 3. Plotting the Splits ---
plt.style.use(['science', 'ieee'])

# Changed to a 2x2 grid to make room for the Raw Data plot
fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
ax_raw, ax_samp, ax_base, ax_corr = axes.flatten()

# 1. Plot Raw Data (Before Cleaning)
ax_raw.plot(df_samp[dist_col] * 1e6, df_samp[force_col] * 1e6, label='Raw Sample', color='black', alpha=0.6, lw=1.5)
ax_raw.plot(df_base[dist_col] * 1e6, df_base[force_col] * 1e6, label='Raw Baseline', color='grey', alpha=0.6, linestyle='--', lw=1.5)
ax_raw.set_title('Raw Data (Includes Air-Phase)')
ax_raw.set_xlabel(r'Displacement (micro m)')
ax_raw.set_ylabel(r'Force (micro N)')
ax_raw.legend(loc='best')

# 2. Plot Cleaned NCM Particle Phases
if not samp_l.empty:
    ax_samp.plot(samp_l[dist_col] * 1e6, samp_l[force_col] * 1e6, label='Loading', color='blue', lw=1.5)
if not samp_b.empty:
    ax_samp.plot(samp_b[dist_col] * 1e6, samp_b[force_col] * 1e6, label='Backlash (Hold)', color='red', lw=2.0)
if not samp_u.empty:
    ax_samp.plot(samp_u[dist_col] * 1e6, samp_u[force_col] * 1e6, label='Unloading', color='green', lw=1.5)

ax_samp.set_title('NCM Particle Phases')
ax_samp.set_xlabel(r'Displacement (micro m)')
ax_samp.set_ylabel(r'Force (micro N)')
ax_samp.legend(loc='best')

# 3. Plot Cleaned Teflon Baseline Phases
if not base_l.empty:
    ax_base.plot(base_l[dist_col] * 1e6, base_l[force_col] * 1e6, label='Loading', color='blue', lw=1.5)
if not base_b.empty:
    ax_base.plot(base_b[dist_col] * 1e6, base_b[force_col] * 1e6, label='Backlash (Hold)', color='red', lw=2.0)
if not base_u.empty:
    ax_base.plot(base_u[dist_col] * 1e6, base_u[force_col] * 1e6, label='Unloading', color='green', lw=1.5)

ax_base.set_title('Teflon Baseline Phases')
ax_base.set_xlabel(r'Displacement (micro m)')
ax_base.set_ylabel(r'Force (micro N)')
ax_base.legend(loc='best')

# 4. Plot Compliance Removed
if not samp_l_corr.empty:
    ax_corr.plot(samp_l_corr[dist_col] * 1e6, samp_l_corr[force_col] * 1e6, label='Loading', color='blue', lw=1.5)
if not samp_u_corr.empty:
    ax_corr.plot(samp_u_corr[dist_col] * 1e6, samp_u_corr[force_col] * 1e6, label='Unloading', color='green', lw=1.5)

ax_corr.set_title('Compliance Removed')
ax_corr.set_xlabel(r'Displacement (micro m)')
ax_corr.set_ylabel(r'Force (micro N)')
ax_corr.legend(loc='best')

plt.tight_layout()
plt.savefig("phase_split_diagnostic.png", bbox_inches='tight')
print("Diagnostic plot saved to 'phase_split_diagnostic.png'")
plt.show()