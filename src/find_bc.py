import logging
import time

from src.sampling import *
from src.constants import *
from src.typings import *
from src.utils import *
from src.linear_problem import *
from src.check_bc import *

logger = logging.getLogger("synthesizeBC")
sampling_logger = logging.getLogger("sampling")

def find_bc(circuit: Circuit,
            type_bc,
            n_samples: int,
            Z : list[sym.Symbol],
            Z0 : list[dict],
            Zu : list[dict],
            deg: int,
            log_level = logging.INFO,
            opt_meth = HIGHSIPM,
            **kwargs):


    logger.setLevel(log_level)
    sampling_logger.setLevel(log_level)
    start = time.perf_counter()

    if type_bc == FINITE_HORIZON:
        required_keys = ["steps"]
        circuit = circuit[0]
    elif type_bc == INFINITE_HORIZON:
        required_keys = ["k", "epsilon","gamma"]
    else:
        raise ValueError("Unknown type_bc")

    for key in required_keys:
        if key not in kwargs:
            raise ValueError("Required key '%s' not provided" % key)


    # Generate variables
    var = generate_variables(Z)

    # Generating terms
    all_terms = generate_terms(var, deg)

    # Sort terms
    all_terms = sort_terms(all_terms)

    init_terms = []
    num_coefficients = None
    q, verification_time, barrier_certificate = None, None, None

    for term in all_terms:
        init_terms.append(term)
        l = len(init_terms)




        # If the chosen bc is finite horizon
        if type_bc == FINITE_HORIZON:
            # Sampling




            steps = int(kwargs["steps"])
            sampling_logger.info("Sampling states")
            i_samples, u_samples, d_samples = sample(Z, n_samples, Z0, Zu, [])
            state_vectors = create_state_vectors(d_samples, Z)
            dynamic_samples = generate_fx_samples(state_vectors, circuit, Z)

            num_coefficients = 2 * l + 3
            c = np.zeros(num_coefficients)
            c[-1] = -1

            bounds = [(-np.inf, np.inf)] * num_coefficients
            y_upper_bound = np.random.uniform(1, 10)
            bounds[-1] = (-y_upper_bound, y_upper_bound)

            i_values, u_values, dynamic_values, d_values = generate_values(init_terms, i_samples, u_samples, d_samples, dynamic_samples)

            # Generate all LP problem constraints
            logger.info("Making constraints for the linear optimization problem...")
            Aub, bub = generate_all_constraints_fin(i_values, u_values, dynamic_values, d_values, steps, True)
            #print(Aub, bub)

            # LP problem
            logger.info("Solveing LP proble...")
            barrier_certificate, a, optimals = (solve_lp_fin(c, Aub, bub, bounds, l, init_terms, var, opt_meth))
            if a is None:
                print("not solved")
                continue
            logger.info("SOLVED")
            # Chek barrier
            q = check_barrier_fin(Z, barrier_certificate, circuit, var, Z0, Zu, optimals[0], optimals[2], optimals[1])




        elif type_bc == INFINITE_HORIZON:
            k = int(kwargs["k"])
            epsilon = float(kwargs["epsilon"])
            gamma = float(kwargs["gamma"])

            num_coefficients = len(circuit) * 2 * l + 1
            c = np.zeros(num_coefficients)
            c[-1] = -1

            bounds = [(-np.inf, np.inf)] * num_coefficients
            y_upper_bound = np.random.uniform(1, 10)
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
            i_samples = sample_states(Z0, Z, n_samples)
            i_sampled_terms = generate_sampled_terms(init_terms, i_samples)
            i_values = separate_real_imag(i_sampled_terms)
            i_values_list.append(i_values)
            sampling_logger.info("Initial states sampled.")

            # create constraints for each barrier function
            for unitary in circuit:
                # second constraint
                sampling_logger.info("Sampling unsafe states...")
                u_samples = sample_states(Zu, Z, n_samples)
                u_sampled_terms = generate_sampled_terms(init_terms, u_samples)
                u_values = separate_real_imag(u_sampled_terms)
                u_values_list.append(u_values)
                sampling_logger.info("Unsafe states sampled")

                # third constraint
                sampling_logger.info("Sampling from the state space Z...")
                d_samples = sample_states([], Z, n_samples)
                state_vectors = create_state_vectors(d_samples, Z)
                dynamic_samples = generate_fx_samples(state_vectors, unitary, Z)
                d_sampled_terms = generate_sampled_terms(init_terms, d_samples)
                dynamic_sampled_terms = generate_sampled_terms(init_terms, dynamic_samples)
                d_values = separate_real_imag(d_sampled_terms)
                dynamic_values = separate_real_imag(dynamic_sampled_terms)
                dynamic_initial_real_imag_list.append(d_values)
                dynamic_real_imag_list.append(dynamic_values)

            # fourth constraint values
            time_samples = sample_states([], Z, n_samples)
            time_sampled_terms = generate_sampled_terms(init_terms, time_samples)
            time_values = separate_real_imag(time_sampled_terms)
            time_real_imag_list.append(time_values)

            # fifth constraint
            k_samples = sample_states([], Z, n_samples)
            state_vectors = create_state_vectors(k_samples, Z)
            k_dynamic_samples = generate_k_fx_samples(state_vectors, circuit * k if len(circuit) < k else circuit, Z)
            k_sampled_terms = generate_sampled_terms(init_terms, k_samples)
            k_dynamic_sampled_terms = generate_sampled_terms(init_terms, k_dynamic_samples)
            k_values = separate_real_imag(k_sampled_terms)
            k_dynamic_values = separate_real_imag(k_dynamic_sampled_terms)
            k_initial_real_imag_list.append(k_values)
            k_real_imag_list.append(k_dynamic_values)
            sampling_logger.info("Sampled from Z.")

            logger.info("Making constraints for the linear optimization problem...")
            Aub, bub = generate_all_constraints_inf(i_values_list,
                                                u_values_list,
                                                dynamic_real_imag_list,
                                                dynamic_initial_real_imag_list,
                                                time_real_imag_list,
                                                k_real_imag_list,
                                                k_initial_real_imag_list,
                                                k,
                                                epsilon,
                                                gamma,
                                                unitaries=circuit)

            logger.info("Constraints made.")
            barrier_certificate, a, y = solve_lp_inf(c, Aub, bub, bounds, l, len(circuit), init_terms, var)
            end = time.perf_counter()
            generation_time = end - start

            if y > k * (epsilon + gamma):
                logger.info("Candidate barrier certificate found")
                for barrier in barrier_certificate: logger.info(str(barrier))
                logger.info("Time for the generation of the candidate: " + str(generation_time))
                start = time.perf_counter()
                q = check_barrier_inf(Z, barrier_certificate, circuit, Z0, Zu, y, epsilon, gamma, k)
                end = time.perf_counter()
                verification_time = end - start
            else:
                logger.info("No valid candidate found.")
                logger.info("Extending template...")
                continue


        if not q:
            logger.info("Barrier certificate is verified by SMT solver")
            logger.info("Time for the verification: " + str(verification_time))
            logger.info(str(barrier_certificate))
            break
        else:
            logger.info("No valid candidate found.")
            logger.info("Extending template...")






























