from scipy.optimize import linprog
import numpy as np
from typing import Tuple
from src.utils import *
from src.log import *
import logging 

logger = logging.getLogger("linearOpt")


def generate_constraints(
    real_imag_list: List[np.ndarray],
    constraint_type: str,
    num_time_steps:int,
    dynamic_real_imag_list: List[np.ndarray] = None,
    right_hand_value: float = 0,
    epsilon: float = 0,
    gamma: float = 0,
    unitaries: List[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate constraints for a linear programming (LP) problem.
    ...
    """
    
    l = len(unitaries)

    if constraint_type == 'initial':
        # Constraints at time t=0
        real_imag = real_imag_list[0]
        re = real_imag[:, :, 0]
        im = real_imag[:, :, 1]
        
        
        num_samples, num_terms = re.shape
        num_vars_per_time = 2 * num_terms  # Real and imaginary parts
        # Prepare Aub
        Aub_coeffs = np.zeros((num_samples, l * num_vars_per_time + 1))
        Aub_coeffs[:, 0:num_vars_per_time] = np.hstack([re, -im])  # B_0(z)

        # Right-hand side
        Aub = np.vstack(Aub_coeffs)
        bub = np.full(num_samples, right_hand_value, dtype=np.float64)

    elif constraint_type == 'unsafe':
        # Constraints for all time steps
        Aub_list = []
        bub_list = []
        for t in range(len(unitaries)):
            real_imag = real_imag_list[t]
            re = real_imag[:, :, 0]
            im = real_imag[:, :, 1]
            num_samples, num_terms = re.shape
            num_vars_per_time = 2 * num_terms  # Real and imaginary parts

            Aub_coeffs = np.zeros((num_samples, l * num_vars_per_time + 1))
            start_idx = t * num_vars_per_time
            Aub_coeffs[:, start_idx:start_idx + num_vars_per_time] = np.hstack([-re, im])  # -B_t(z)

            # Append 1 for 'y' variable
            Aub_coeffs[:, -1] = 1.0

            Aub_list.append(Aub_coeffs)
            bub_list.append(np.zeros(num_samples, dtype=np.float64))

        Aub = np.vstack(Aub_list)
        bub = np.hstack(bub_list)
    elif constraint_type == 'dynamic':
        # Constraints involving dynamics: B_t(f_t(z)) - B_t(z) <= epsilon
        Aub_list = []
        bub_list = []
        for t in range(len(unitaries)):
            real_imag_t = real_imag_list[t]
            dynamic_real_imag_t = dynamic_real_imag_list[t]

            delta = dynamic_real_imag_t - real_imag_t
            delta_re = delta[:, :, 0]
            delta_im = delta[:, :, 1]
            num_samples, num_terms = delta_re.shape
            num_vars_per_time = 2 * num_terms  # Real and imaginary parts

            Aub_coeffs = np.zeros((num_samples, l * num_vars_per_time + 1))
            start_idx = t * num_vars_per_time
            Aub_coeffs[:, start_idx:start_idx + num_vars_per_time] = np.hstack([delta_re, -delta_im])  # B_t(f_t(z)) - B_t(z)

            Aub_list.append(Aub_coeffs)
            bub_list.append(np.full(num_samples, epsilon, dtype=np.float64))

        Aub = np.vstack(Aub_list)
        bub = np.hstack(bub_list)

    elif constraint_type == 'time_evolution':
        # Constraints involving time evolution: |B_{t+1}(z) - B_t(z)| <= gamma
        Aub_list = []
        bub_list = []
        real_imag = real_imag_list[0]
        re = real_imag[:, :, 0]
        im = real_imag[:, :, 1]
        num_samples, num_terms = re.shape
        num_vars_per_time = 2 * num_terms  
        
        for sign in [1, -1]: 
            Aub_coeffs = np.zeros((num_samples, l * num_vars_per_time + 1))
            start_idx_t = 0
            start_idx_t1 = num_vars_per_time

            # Coefficients for B_{t+1}(z)
            Aub_coeffs[:, start_idx_t1:start_idx_t1 + num_vars_per_time] = sign * np.hstack([re, -im])

            # Subtract coefficients for B_t(z)
            Aub_coeffs[:, start_idx_t:start_idx_t + num_vars_per_time] -= sign * np.hstack([re, -im])

            # Right-hand side
            bub = np.full(num_samples, gamma, dtype=np.float64)

            Aub_list.append(Aub_coeffs)
            bub_list.append(bub)

        Aub = np.vstack(Aub_list)
        bub = np.hstack(bub_list)


    elif constraint_type == 'k_inductive':
        # Constraints for k-induction: B_{t+k}(f^{(k)}(z)) - B_t(z) <= 0
        Aub_list = []
        bub_list = []
        real_imag_tk = real_imag_list[0]
        real_imag_t = dynamic_real_imag_list[0]

        delta = real_imag_tk - real_imag_t
        delta_re = delta[:, :, 0]
        delta_im = delta[:, :, 1]
        num_samples, num_terms = delta_re.shape
        num_vars_per_time = 2 * num_terms  # Real and imaginary parts

        Aub_coeffs = np.zeros((num_samples, l * num_vars_per_time + 1))
        start_idx = 0
        Aub_coeffs[:, start_idx:start_idx + num_vars_per_time] = np.hstack([delta_re, -delta_im])  # B_{t+k}(f^{(k)}(z))

        Aub_list.append(Aub_coeffs)
        bub_list.append(np.full(num_samples, right_hand_value, dtype=np.float64))

        Aub = np.vstack(Aub_list)
        bub = np.hstack(bub_list)

    else:
        raise ValueError("Invalid constraint_type.")

    return Aub, bub


def generate_all_constraints(
    initial_real_imag_list,
    unsafe_real_imag_list,
    dynamic_real_imag_list,
    dynamic_initial_real_imag_list,
    time_real_imag_list,
    k_real_imag_list,
    k_initial_real_imag_list,
    K: int,
    epsilon: float,
    gamma: float,
    right_hand_value: float = 0,
    unitaries: List[np.ndarray] = None
):
    """
    Generate all constraints for the LP problem.
    ...
    """
    # Generate initial constraints
    Aub_initial, bub_initial = generate_constraints(
        real_imag_list=initial_real_imag_list,
        constraint_type='initial',
        unitaries=unitaries,
        num_time_steps=K,
        right_hand_value=right_hand_value
    )

    # Generate unsafe constraints
    Aub_unsafe, bub_unsafe = generate_constraints(
        real_imag_list=unsafe_real_imag_list,
        constraint_type='unsafe',
        num_time_steps=K,
        right_hand_value=right_hand_value,
        unitaries=unitaries
    )

    # Generate dynamic constraints
    Aub_dynamic, bub_dynamic = generate_constraints(
        real_imag_list=dynamic_initial_real_imag_list,
        dynamic_real_imag_list=dynamic_real_imag_list,
        constraint_type='dynamic',
        num_time_steps=K,
        epsilon=epsilon,
        unitaries=unitaries
    )

    # Generate k-inductive constraints
    Aub_k_inductive, bub_k_inductive = generate_constraints(
        real_imag_list=k_real_imag_list,
        dynamic_real_imag_list=k_initial_real_imag_list,
        constraint_type='k_inductive',
        unitaries=unitaries,
        num_time_steps=K,
        right_hand_value=right_hand_value
    )

    # Concatenate all inequality constraints
    Aub = np.vstack([Aub_initial, Aub_unsafe, Aub_k_inductive])
    bub = np.hstack([bub_initial, bub_unsafe, bub_k_inductive])
    if K > 1: 
        Aub = np.vstack([Aub, Aub_dynamic])
        bub = np.hstack([bub, bub_dynamic])
    
    if len(unitaries) > 1:
        # Generate time evolution constraints
        Aub_time_evolution, bub_time_evolution = generate_constraints(
            real_imag_list=time_real_imag_list,
            constraint_type='time_evolution',
            num_time_steps=K,
            gamma=gamma,
            unitaries=unitaries
        )
        Aub = np.vstack([Aub, Aub_time_evolution])
        bub = np.hstack([bub, bub_time_evolution])
    
    return Aub, bub


def solve_lp(c, Aub, bub, bounds, l, num_time_steps, terms, vars, log_level=logging.INFO):
    logger.info("Solving linear optimization problem...")
    result = linprog(
        c,
        A_ub=Aub,
        b_ub=bub,
        bounds=bounds,
        method='highs'
    )
    logger.info("Linear program terminated.")
    logger.setLevel(log_level)
    if result.success:
        logger.info("Optimization completed:" + result.message)
        #x_optimal = round_values(result.x)
        x_optimal = result.x
        y_optimal = x_optimal[-1]
        logger.debug(result.x)
        logger.info("Optimal value: " + str(y_optimal))

        # Extract coefficients for each time step
        a_optimal = []
        for t in range(num_time_steps):
            start_idx = t * 2 * l
            end_idx = start_idx + l
            Re_a_optimal = x_optimal[start_idx:end_idx]
            Im_a_optimal = x_optimal[end_idx:end_idx+l]
            a_t = construct_complex_coefficients(Re_a_optimal, Im_a_optimal)
            a_optimal.append(a_t)
            logger.info("Extracting the real part of coefficients a form result...")
            logger.info(str(Re_a_optimal))
            logger.info("Extracting the immaginary part of coefficients a form result...")
            logger.info(str(Im_a_optimal))

        # Generate barrier polynomials for each time step
        barrier_certificates = []
        for t in range(num_time_steps):
            barrier_certificate = generate_barrier_polynomial(a_optimal[t], terms, vars)
            barrier_certificates.append(barrier_certificate)

        return barrier_certificates, a_optimal, y_optimal

    else:
        logger.warning("Optimization falied: " + result.message)
        #print("Optimization failed:", result.message)
        return None, None, None
