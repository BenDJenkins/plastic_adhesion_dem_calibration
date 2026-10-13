import numpy as np

def calculate_eepa_force(d, E_star, R_star, m, lam_p, chi, F_0, d0, gamma_star):
    """
    Calculates the EEPA normal force step-by-step matching the conditional 
    branching logic and dynamic cache state updates.
    """
    n_points = len(d)
    forces = np.zeros(n_points)
    
    k_1 = (4.0 / 3.0) * E_star * np.sqrt(R_star)
    k_2 = k_1 / (1.0 - lam_p)
    
    inv_m = 1.0 / m
    lam_p_inv_m = lam_p * inv_m
    m_over_lam_p = m / lam_p
    adhesion_coeff = 1.5 * np.pi * gamma_star
    chi_over_m = chi * inv_m if m != 0 else 0.0
    
    max_overlap_history = 0.0

    for i in range(n_points):
        delta = d[i] - d0
        if delta <= 0:
            forces[i] = 0.0
            continue
            
        max_overlap_history = max(max_overlap_history, delta)
        
        delta_m = delta ** m
        f_1 = k_1 * delta_m
        
        delta_p = lam_p_inv_m * max_overlap_history
        delta_p_m = delta_p ** m
        f_2 = k_2 * (delta_m - delta_p_m)
        
        if f_2 >= f_1:
            forces[i] = F_0 + f_1
            continue
            
        delta_chi = delta ** chi
        alpha = np.sqrt(2.0 * delta_p * R_star)
        
        F_min_raw = F_0 - adhesion_coeff * alpha
        F_min_limit = F_0 - k_2 * delta_p_m
        F_min = max(F_min_raw, 0.5 * (F_0 + F_min_limit))
        
        delta_term = delta_p_m + (F_min - F_0) / k_2
        k_a = (F_min - F_0) / (delta_term ** chi_over_m) if (delta_term > 0 and chi > 0) else 0.0
        f_a = k_a * delta_chi
        
        if f_1 > f_2 > f_a:
            forces[i] = F_0 + f_2
            continue
            
        if f_a > f_2:
            base_term = max(0.0, delta_m - f_a / k_2)
            delta_max_new = (base_term ** inv_m) * m_over_lam_p
            max_overlap_history = delta_max_new
            forces[i] = F_0 + f_a
            continue
            
    return forces


def evaluate_eepa_single(d, max_overlap_history, E_star, R_star, m=1.0, lam_p=0.0, chi=1.0, F_0=0.0, d0=0.0, gamma_star=0.0):
    """
    Evaluates the EEPA force for a single displacement point.
    Returns: (force, updated_max_overlap)
    """
    delta = d - d0
    if delta <= 0:
        return 0.0, max_overlap_history
        
    k_1 = (4.0 / 3.0) * E_star * np.sqrt(R_star)
    k_2 = k_1 / (1.0 - lam_p)
    
    inv_m = 1.0 / m
    lam_p_inv_m = lam_p * inv_m
    m_over_lam_p = m / lam_p
    chi_over_m = chi * inv_m if m != 0 else 0.0
    adhesion_coeff = 1.5 * np.pi * gamma_star
    
    # Track maximum overlap encountered
    current_max_overlap = max(max_overlap_history, delta)
    
    delta_m = delta ** m
    f_1 = k_1 * delta_m
    
    delta_p = lam_p_inv_m * current_max_overlap
    delta_p_m = delta_p ** m
    f_2 = k_2 * (delta_m - delta_p_m)
    
    # Branch 1
    if f_2 >= f_1:
        return F_0 + f_1, current_max_overlap
        
    delta_chi = delta ** chi
    alpha = np.sqrt(2.0 * delta_p * R_star)
    
    F_min_raw = F_0 - adhesion_coeff * alpha
    F_min_limit = F_0 - k_2 * delta_p_m
    F_min = max(F_min_raw, 0.5 * (F_0 + F_min_limit))
    
    delta_term = delta_p_m + (F_min - F_0) / k_2
    k_a = (F_min - F_0) / (delta_term ** chi_over_m) if (delta_term > 0 and chi > 0) else 0.0
    f_a = k_a * delta_chi
    
    # Branch 2
    if f_1 > f_2 > f_a:
        return F_0 + f_2, current_max_overlap
        
    # Branch 3 (Plastic/Adhesive update)
    if f_a > f_2:
        base_term = max(0.0, delta_m - f_a / k_2)
        delta_max_new = (base_term ** inv_m) * m_over_lam_p
        return F_0 + f_a, delta_max_new
        
    return 0.0, current_max_overlap
