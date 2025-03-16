import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.utils import *
from src.sampling import *
from src.k_optimization import *
from src.gates import *
import sympy as sym
from src.check_barrier import *
from src.log import *
import logging

logger = logging.getLogger("synthesizeBC")
sampling_logger = logging.getLogger("sampling")

def find_barrier(Z, var, unitaries, N_SAMPLES, Z0, Zu, gamma=0, epsilon=0, k=1, DEG=2, log_level=logging.INFO):
    logger.setLevel(log_level)
    sampling_logger.setLevel(log_level)
    start = time.perf_counter()
    logger.info("Generating polynomial terms...")
    all_terms = generate_terms(var, DEG)
    logger.info("Polynomial terms generated.")
    logger.info(str(all_terms))
    init_terms = [(Z[0], Z[0].conjugate())]
    init_terms.append(all_terms[0])
    terms_toadd = [term for term in all_terms if term not in init_terms]
    iteration = 1

    for term in terms_toadd:
        logger.info('-------------------iteration ' + str(iteration) + '----------------------')
        logger.info("Setting template terms")
        logger.info(str(init_terms))
        l = len(init_terms)
        num_coefficients = len(unitaries)*2 * l + 1
        c = np.zeros(num_coefficients)
        c[-1] = -1
        bounds = [(-np.inf, np.inf)] * num_coefficients
        y_upper_bound = np.random.uniform(1,10)
        bounds[-1] = (-y_upper_bound, y_upper_bound)

        i_values_list = []
        u_values_list = []
        dynamic_initial_real_imag_list = []
        dynamic_real_imag_list = []
        time_real_imag_list = []
        k_real_imag_list = []
        k_initial_real_imag_list = []

        # First constraint
        sampling_logger.info("Sampling initial states...")
        i_samples = sample_states_qmc2(Z0, Z, N_SAMPLES)
        i_sampled_terms = generate_sampled_terms(init_terms, i_samples)
        i_values = separate_real_imag(i_sampled_terms)
        i_values_list.append(i_values)
        sampling_logger.info("Initial states sampled.")

        # create constraints for each barrier function
        for unitary in unitaries:
            # second constraint
            sampling_logger.info("Sampling unsafe states...")
            u_samples = sample_states_qmc2(Zu,Z, N_SAMPLES)
            u_sampled_terms = generate_sampled_terms(init_terms, u_samples)
            u_values = separate_real_imag(u_sampled_terms)
            u_values_list.append(u_values)
            sampling_logger.info("Unsafe states sampled")
            
            # third constraint
            sampling_logger.info("Sampling from the state space Z...")
            d_samples = sample_states_qmc2([], Z, N_SAMPLES)
            state_vectors = create_state_vectors(d_samples, Z)
            dynamic_samples = generate_fx_samples(state_vectors, unitary, Z)
            d_sampled_terms = generate_sampled_terms(init_terms, d_samples)
            dynamic_sampled_terms = generate_sampled_terms(init_terms, dynamic_samples)    
            d_values = separate_real_imag(d_sampled_terms)
            dynamic_values = separate_real_imag(dynamic_sampled_terms)
            dynamic_initial_real_imag_list.append(d_values)
            dynamic_real_imag_list.append(dynamic_values)
            
        # fourth constraint values
        time_samples = sample_states_qmc2([], Z, N_SAMPLES)
        time_sampled_terms = generate_sampled_terms(init_terms, time_samples)
        time_values = separate_real_imag(time_sampled_terms)
        time_real_imag_list.append(time_values)   
            
        # fifth constraint
        k_samples = sample_states_qmc2([], Z, N_SAMPLES)
        state_vectors = create_state_vectors(k_samples, Z)
        k_dynamic_samples = generate_k_fx_samples(state_vectors, unitaries*k if len(unitaries)<k else unitaries, Z)
        k_sampled_terms = generate_sampled_terms(init_terms, k_samples)
        k_dynamic_sampled_terms = generate_sampled_terms(init_terms, k_dynamic_samples)    
        k_values = separate_real_imag(k_sampled_terms)
        k_dynamic_values = separate_real_imag(k_dynamic_sampled_terms)
        k_initial_real_imag_list.append(k_values)
        k_real_imag_list.append(k_dynamic_values)
        sampling_logger.info("Sampled from Z.")

        logger.info("Making constraints for the linear optimization problem...")
        Aub, bub = generate_all_constraints(i_values_list, 
                                            u_values_list, 
                                            dynamic_real_imag_list, 
                                            dynamic_initial_real_imag_list, 
                                            time_real_imag_list, 
                                            k_real_imag_list, 
                                            k_initial_real_imag_list,
                                            k, 
                                            epsilon, 
                                            gamma,                                
                                            unitaries = unitaries)
        logger.info("Constraints made.")
        barrier_certificate, a, y = solve_lp(c, Aub, bub, bounds, l, len(unitaries), init_terms, var)
        end = time.perf_counter()
        generation_time = end-start
            
        if y > k*(epsilon + gamma):
            logger.info("Candidate barrier certificate found")
            for barrier in barrier_certificate: logger.info(str(barrier))
            logger.info("Time for the generation of the candidate: " + str(generation_time))
            start = time.perf_counter()
            q = check_barrier(Z, barrier_certificate, unitaries, Z0, Zu, y, epsilon, gamma, k)
            end = time.perf_counter()
            verification_time = end-start
            if not q: 
                logger.info("Barrier certificate is verified by SMT solver")
                logger.info("Time for the verification: " + str(verification_time))
                logger.info(str(barrier_certificate))
                break
        else:
            logger.info("No valid candidate found.")
            logger.info("Extending template...")
        init_terms.append(term)
        iteration += 1