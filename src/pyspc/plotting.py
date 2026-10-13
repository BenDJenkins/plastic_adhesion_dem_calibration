import matplotlib.pyplot as plt
from .force_data_preprocessing import preprocess_force_data
from .force_balance_indentor import force_balance_indentor, run_optimization, calculate_eepa_force
import numpy as np
import io
import base64


def plot_test(df_l, df_u, save_name='test.png'):   

    plt.plot(df_l['indentation_um'], df_l['force_nN'], label='Loading')
    # plt.plot(df_b['indentation_um'], df_b['force_nN'], label='Backlash')
    plt.plot(df_u['indentation_um'], df_u['force_nN'], label='Unloading')
    plt.xlabel('Indentation (µm)')
    plt.ylabel('Force (nN)')
    plt.title('Indentation-Force Curve')
    plt.legend()
    plt.savefig(save_name, bbox_inches='tight')
    print("Plot saved as 'test.png'")


def plot_eepa_fit(df_load, df_unload, df_eepa, eepa_params=None,
                  force_col='force_nN', dist_col='indentation_um', 
                  output_image_name='eepa_fit_results.png'):
    """Generates and saves the IEEE-formatted plot."""

    full_dist = np.concatenate([df_load[dist_col].values, df_unload[dist_col].values])

    _, eepa_model_forces = df_eepa

    plt.style.use(['science', 'ieee'])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    
    # Assuming 'distance' is in meters based on the previous 1e6 multiplier
    ax.scatter(df_load[dist_col] * 1e6, df_load[force_col], 
               s=2, color='black', alpha=0.35, label='Exp: Loading')
    ax.scatter(df_unload[dist_col] * 1e6, df_unload[force_col], 
               s=2, color='black', alpha=0.35, label='Exp: Unloading')
    ax.scatter(full_dist * 1e6, eepa_model_forces, 
               s=2, color='green', alpha=0.35, label='EEPA Model Fit')
    
    ax.set_xlabel(r'Displacement ($\mu$m)')
    ax.set_ylabel(r'Force (N)')
    ax.set_title(r'EEPA Split Loading \& Unloading Fit ($m=1.0$)')

    if eepa_params is not None:
        E_fit, lp_fit = eepa_params
        E_display = f"{E_fit/1e9:.2f} GPa" if E_fit >= 1e9 else f"{E_fit/1e6:.2f} MPa" if E_fit >= 1e6 else f"{E_fit/1e3:.2f} kPa"
        legend_text = f'$E^*={E_display}$\n$m=1.00$\n$\\lambda_p={lp_fit:.2f}$'
        ax.legend(title=legend_text, frameon=True, loc='best')
    
    plt.savefig(output_image_name, bbox_inches='tight')
    print(f"Plot successfully saved to {output_image_name}")
    return fig


def create_base64_plot(df_load, df_unload, model_forces, params):
    """Creates a matplotlib plot and returns a base64 encoded string."""
    E_fit, lp_fit = params
    fig = plot_eepa_fit(df_load, df_unload, model_forces, (E_fit, lp_fit))
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
    buf.seek(0)
    
    # Encode to base64
    return base64.b64encode(buf.read()).decode('utf-8')
