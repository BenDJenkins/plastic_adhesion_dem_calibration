from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import Dict
import scienceplots
from scipy.optimize import differential_evolution
from scipy.optimize import brentq
from .utilities import generate_needle_properties, generate_particle_properties, generate_optimization_settings, calculate_r_star, generate_substrate_properties, calculate_e_star
from .contact_models.eepa import calculate_eepa_force, evaluate_eepa_single


def force_balance_indentor(displacement_array, particle_radius: float, 
                           particle_youngs_modulus: float, particle_plasticity_ratio: float, 
                           params_particle: dict[str, float], params_needle: dict[str, float], params_substrate: dict[str, float]):
    """
    Simulates the force balance simluation of a particle compressed between a needle and a stage.
    
    Parameters:
    - displacement_array: Array of stage displacements.
    - particle_radius: Radius of the simulated particle.
    - params_needle: Dictionary of EEPA parameters for the needle.
    - params_substrate: Dictionary of EEPA parameters for the substrate.
    """
    r_star = calculate_r_star(particle_radius)
    e_star_particle_needle = calculate_e_star(particle_youngs_modulus, params_particle['poisson_ratio'], 
                                              params_substrate['youngs_modulus'], params_substrate['poisson_ratio'])
    e_star_particle_substrate = calculate_e_star(particle_youngs_modulus, params_particle['poisson_ratio'], 
                                                 params_needle['youngs_modulus'], params_needle['poisson_ratio'])

    n_points = len(displacement_array)
    force_array = np.zeros(n_points)
    true_needle_overlaps = np.zeros(n_points)
    
    # Track the histories independently for both interfaces
    max_overlap_needle = 0.0
    max_overlap_stage = 0.0
    
    needle_position = particle_radius

    for i, displacement in enumerate(displacement_array):
        substrate_position = -particle_radius + displacement  
        
        def force_balance(x):
            """Calculates net force on the particle at position x."""
            n_overlap = max(0.0, (x + particle_radius) - needle_position)
            s_overlap = max(0.0, substrate_position - (x - particle_radius))
            
            # During trial guesses, we only want the force [0], not the updated history
            f_n, _ = evaluate_eepa_single(n_overlap, max_overlap_needle, e_star_particle_needle, r_star, lam_p=particle_plasticity_ratio)
            f_s, _ = evaluate_eepa_single(s_overlap, max_overlap_stage, e_star_particle_substrate, r_star, lam_p=particle_plasticity_ratio)
            
            return f_s - f_n
            
        # Skip root finding if there is no macroscopic contact yet
        if displacement <= 0:
            particle_position = 0.0
        else:
            # Find equilibrium particle position (where net force is 0)
            try:
                particle_position = brentq(force_balance, substrate_position, needle_position)
            except ValueError:
                # Fallback if brentq fails (usually means one of the bounds is perfectly 0)
                particle_position = substrate_position if abs(force_balance(substrate_position)) < abs(force_balance(needle_position)) else needle_position

        # Calculate final state overlaps at the confirmed equilibrium position
        n_overlap_final = max(0.0, (particle_position + particle_radius) - needle_position)
        s_overlap_final = max(0.0, substrate_position - (particle_position - particle_radius))
        
        # Calculate final force AND retrieve the updated history states
        final_force, max_overlap_needle = evaluate_eepa_single(n_overlap_final, max_overlap_needle, e_star_particle_needle, r_star, lam_p=particle_plasticity_ratio)
        _, max_overlap_stage = evaluate_eepa_single(s_overlap_final, max_overlap_stage, e_star_particle_substrate, r_star, lam_p=particle_plasticity_ratio)
        
        # Save data
        force_array[i] = final_force
        true_needle_overlaps[i] = n_overlap_final
        
    return force_array, true_needle_overlaps


def indentor_objective_function(optisation_parameters, load_dist, load_force, unload_dist, unload_force, parameters):
    """Evaluates combined RMSE across loading and unloading trajectories."""

    particle_properties, params_needle, params_substrate = parameters

    # Unpack optimization parameters
    youngs_modulus, plasticiity_ratio = optisation_parameters

    # Get full set of displacement data
    full_dist = np.concatenate([load_dist, unload_dist])

    # Run the force balance simulation and get the predicted forces for this set of Young's modulus and plasticity ratio
    force_pred, _ = force_balance_indentor(full_dist, particle_properties['radius'], 
                                           youngs_modulus, plasticiity_ratio, 
                                           particle_properties, params_needle, params_substrate)
    
    pred_load = force_pred[:len(load_dist)]
    pred_unload = force_pred[len(load_dist):]

    rmse_load = np.sqrt(np.mean((pred_load - load_force)**2))
    rmse_unload = np.sqrt(np.mean((pred_unload - unload_force)**2))

    return 0.5 * (rmse_load + rmse_unload)


def run_optimization(
        df_load, df_unload,
        particle_properties: dict[str, float], optimization_settings=None,
        params_needle=None, params_substrate=None,
        force_col='force_nN', dist_col='indentation_um'
        ) -> tuple[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    
    """Configures and runs the differential evolution optimizer."""
    if particle_properties is None:
        raise ValueError("particle_properties must be provided.")

    if optimization_settings is None:
        optimization_settings = generate_optimization_settings()

    if params_needle is None:
        params_needle = generate_needle_properties()

    if params_substrate is None:
        params_substrate = generate_substrate_properties()

    parameters = (particle_properties, params_needle, params_substrate)

    # Seperate out indentation displacement and force values for loading and unloading
    load_d, load_f = df_load[dist_col].values, df_load[force_col].values
    unload_d, unload_f = df_unload[dist_col].values, df_unload[force_col].values

    # Connect the loading and unloading distances and find the minimum indentation value
    full_dist = np.concatenate([load_d, unload_d])

    # Set the bounds of the 
    bounds = [particle_properties['youngs_modulus_bounds'], 
              particle_properties['plasticity_ratio_bounds']] # [E_star, lam_p]

    # Calibration cycle
    result = differential_evolution(
        indentor_objective_function, 
        bounds, 
        args=(load_d, load_f, unload_d, unload_f, parameters),
        maxiter=5000,
        popsize=optimization_settings['popsize'], 
        tol=optimization_settings['tol'], 
        workers=optimization_settings['workers']
    )

    # Get final EEPA force array
    full_dist = np.concatenate([load_d, unload_d])
    full_forces, _ = force_balance_indentor(full_dist, particle_properties['radius'], 
                                            result.x[0], result.x[1], 
                                            particle_properties, params_needle, params_substrate)

    return result.x, (full_dist, full_forces)
