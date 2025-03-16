import numpy as np
import sympy as sym
import itertools
from typing import Tuple
from scipy.optimize import linprog


'''
Given the number of the qubits, generate the symbols
'''
def generate_symbols(n : int) -> list[sym.Symbol]:
    return [sym.Symbol('z' + str(i), complex=True) for i in range(2**n)]

def generate_variables(Z: list[sym.Symbol]) -> list[sym.Symbol]:
    variables = Z + [z.conjugate() for z in Z]
    return variables

'''
Given the variables we generate all possible terms for the polynomial
'''
def generate_terms(variables : list[sym.Symbol], deg=2) -> list[tuple]:
    p = []
    for d in range(deg + 1):
        p += itertools.combinations_with_replacement(variables, d) # get all possible combinations of variables
    return p

'''
Substitute the terms with the corresponding sampled values values
'''
def generate_sampled_terms(terms : list[tuple], samples) -> list[list]:
    sampled_terms = []
    for sample in samples:
        p = [map(lambda x: sample.get(x, x), term) for term in terms]
        p = [np.prod([t for t in term]) for term in p] # each term is a list of variables, so get the product
        sampled_terms.append(p)
    return sampled_terms


'''
Given the sampled term's values, separate into real and immaginary parts
'''
def separate_real_imag(sampled_terms, num_digit=5) -> list[list[tuple]]:
    values = [[(round(value.real, num_digit),round(value.imag, num_digit)) for value in sample] for sample in sampled_terms]
    return np.array(values)

'''
Get Re(a_i) and Im(a_i) as the values we are seeking
'''
def create_coefficients(p, coeff_real_tok='Re(a', coeff_imag_tok='Im(a'):
    real_coeff = [sym.Symbol(coeff_real_tok + str(i) + ')', complex=True) for i in range(len(p))]
    imag_coeff = [sym.Symbol(coeff_imag_tok + str(i) + ')', complex=True) for i in range(len(p))]
    return real_coeff, imag_coeff

def apply_op(sample_state, op, grover=True):



    state = np.dot(op, sample_state)

    return state

def generate_fx_samples_k_times(state_vectors, op, Z, K):
    dynamic_samples = []
    for state_vector in state_vectors:
        sample = {}
        new_state = state_vector
        for _ in range(K):
            new_state = apply_op(new_state, op)  # Applica l'operatore K volte
        for i, z in enumerate(Z):
            sample[z] = new_state[i]
            sample[sym.conjugate(z)] = np.conj(new_state[i])
        dynamic_samples.append(sample)
    return dynamic_samples

def generate_fx_samples(state_vectors, op, Z):
    dynamic_samples = []
    for state_vector in state_vectors:
        sample = {}
        new_state = apply_op(state_vector, op)
        for i, z in enumerate(Z):
            sample[z] = new_state[i]
            sample[sym.conjugate(z)] = np.conj(new_state[i])
        dynamic_samples.append(sample)
    return dynamic_samples

def generate_constraints(
    real_imag: np.ndarray,
    constraint_type: str,
    dynamic_real_imag: np.ndarray = None,
    right_hand_value: float = 0,
    eps = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate constraints for a linear programming (LP) problem.

    Parameters:
    -----------
    real_imag : np.ndarray
        A NumPy array of shape (num_samples, num_terms, 2), where:
        - num_samples is the number of samples.
        - num_terms is the number of terms in each sample.
        - The last dimension contains the real and imaginary parts (Re, Im).
        Represents the polynomial evaluated at certain points.

    constraint_type : str
        The type of constraint to generate.
        - 'initial': constraints for initial samples
        - 'unsafe': constraints for unsafe samples
        - 'dynamic': constraints involving dynamic_real_imag

    dynamic_real_imag : np.ndarray, optional
        Required if constraint_type is 'dynamic'.
        A NumPy array of the same shape as real_imag.
        Represents the dynamic polynomial evaluated at the same points.

    right_hand_value : float, optional
        The right-hand side value for the inequality constraints (default is 0).

    Returns:
    --------
    Aub : np.ndarray
        The inequality constraint matrix (A_ub) for the LP problem.

    bub : np.ndarray
        The inequality constraint vector (b_ub) for the LP problem.

    Aeq : np.ndarray
        The equality constraint matrix (A_eq) for the LP problem.

    beq : np.ndarray
        The equality constraint vector (b_eq) for the LP problem.
    """
    if constraint_type == 'initial':
        # Extract real and imaginary parts
        re = real_imag[:, :, 0]
        im = real_imag[:, :, 1]
        num_samples, num_terms = re.shape

        # Prepare coefficients for the inequality constraints (Aub)
        # For each sample, interleave re and -im
        Aub_coeffs = np.empty((num_samples, 2 * num_terms), dtype=np.float64)
        #print(f'Aub_coeffs{Aub_coeffs}')
        Aub_coeffs[:, 0::2] = re
        Aub_coeffs[:, 1::2] = -im

        # Append zero for the 'epsilon' variable
        Aub = np.hstack([Aub_coeffs, np.zeros((num_samples, 1))])

        # Append zero for the 'y' variable
        Aub = np.hstack([Aub, np.zeros((num_samples, 1))])

        # Prepare coefficients for the equality constraints (Aeq)
        # For each sample, interleave im and re
        Aeq_coeffs = np.empty((num_samples, 2 * num_terms), dtype=np.float64)
        Aeq_coeffs[:, 0::2] = im
        Aeq_coeffs[:, 1::2] = re

        # Append zero for the 'y' variable
        Aeq = np.hstack([Aeq_coeffs, np.zeros((num_samples, 1))])

        # Right-hand side vectors
        bub = np.full(num_samples, right_hand_value, dtype=np.float64)
        beq = np.zeros(num_samples, dtype=np.float64)

    elif constraint_type == 'unsafe':
        # Extract real and imaginary parts
        re = real_imag[:, :, 0]
        im = real_imag[:, :, 1]
        num_samples, num_terms = re.shape

        # Prepare coefficients for the inequality constraints (Aub)
        # For each sample, interleave -re and im
        Aub_coeffs = np.empty((num_samples, 2 * num_terms), dtype=np.float64)
        Aub_coeffs[:, 0::2] = -re
        Aub_coeffs[:, 1::2] = im


        # Append zero for the 'epsilon' variable
        Aub = np.hstack([Aub_coeffs, np.zeros((num_samples, 1))])

        # Append one for the 'y' variable (since the inequality involves y)
        Aub = np.hstack([Aub, np.ones((num_samples, 1))])

        # Prepare coefficients for the equality constraints (Aeq)
        # For each sample, interleave im and re
        Aeq_coeffs = np.empty((num_samples, 2 * num_terms), dtype=np.float64)
        Aeq_coeffs[:, 0::2] = im
        Aeq_coeffs[:, 1::2] = re

        # Append one for the 'y' variable
        Aeq = np.hstack([Aeq_coeffs, np.zeros((num_samples, 1))])

        # Right-hand side vectors
        bub = np.full(num_samples,0 , dtype=np.float64)
        beq = np.zeros(num_samples, dtype=np.float64)

    elif constraint_type == 'dynamic':
        if dynamic_real_imag is None:
            raise ValueError("dynamic_real_imag must be provided for 'dynamic' constraint_type.")

        # Calculate the difference between dynamic and original values
        delta = dynamic_real_imag  - real_imag  # Shape: (num_samples, num_terms, 2)

        # Separate the real and imaginary parts of the differences
        delta_re = delta[:, :, 0]  # Real parts
        delta_im = delta[:, :, 1]  # Imaginary parts
        num_samples, num_terms = delta_re.shape

        # Prepare coefficients for the inequality constraints (Aub)
        # For each sample, interleave delta_re and -delta_im
        Aub_coeffs = np.empty((num_samples, 2 * num_terms), dtype=np.float64)
        Aub_coeffs[:, 0::2] = delta_re
        Aub_coeffs[:, 1::2] = -delta_im

        if eps:
            # Append ones for the 'epsilon' variable
            Aub = np.hstack([Aub_coeffs, -1 * np.ones((num_samples, 1))])
        else:
            # Append zero for the 'epsilon' variable
            Aub = np.hstack([Aub_coeffs, np.zeros((num_samples, 1))])

        # Append zero for the 'y' variable
        Aub = np.hstack([Aub, np.zeros((num_samples, 1))])

        # Prepare coefficients for the equality constraints (Aeq)
        # For each sample, interleave delta_im and delta_re
        Aeq_coeffs = np.empty((num_samples, 2 * num_terms), dtype=np.float64)
        Aeq_coeffs[:, 0::2] = delta_im
        Aeq_coeffs[:, 1::2] = delta_re

        # Append zero for the 'y' variable
        Aeq = np.hstack([Aeq_coeffs, np.zeros((num_samples, 1))])

        # Right-hand side vectors
        bub = np.full(num_samples, right_hand_value, dtype=np.float64)
        beq = np.zeros(num_samples, dtype=np.float64)

    else:
        raise ValueError("Invalid constraint_type. Must be 'initial', 'unsafe', or 'dynamic'.")

    return Aub, bub, Aeq, beq


def generate_all_constraints(
    initial_real_imag,
    unsafe_real_imag,
    dynamic_real_imag,
    dynamic_initial_real_imag,
    dynamic_real_k_imag,
    k,
    constr = False,
    right_hand_value: float = 0
):
    """
    Generate all constraints for the LP problem.

    Parameters:
    -----------
    initial_real_imag : np.ndarray
        Real and imaginary parts of the initial samples.

    unsafe_real_imag : np.ndarray
        Real and imaginary parts of the unsafe samples.

    dynamic_real_imag : np.ndarray
        Real and imaginary parts of the dynamic samples (after applying dynamics).

    dynamic_initial_real_imag : np.ndarray
        Real and imaginary parts of the initial samples (before applying dynamics).

    right_hand_value : float, optional
        Right-hand side value for inequality constraints (default is 0).

    Returns:
    --------
    Aub : np.ndarray
        Combined inequality constraint matrix.

    bub : np.ndarray
        Combined inequality constraint vector.

    Aeq : np.ndarray
        Combined equality constraint matrix.

    beq : np.ndarray
        Combined equality constraint vector.
    """
    vmatr = []
    hmatr = []
    # Generate initial constraints
    if initial_real_imag is not None:
        Aub_initial, bub_initial, Aeq_initial, beq_initial = generate_constraints(
            real_imag=initial_real_imag,
            constraint_type='initial',
            right_hand_value=right_hand_value
        )

        vmatr.append(Aub_initial)
        hmatr.append(bub_initial)

    # Generate unsafe constraints
    if unsafe_real_imag is not None:
        Aub_unsafe, bub_unsafe, Aeq_unsafe, beq_unsafe = generate_constraints(
            real_imag=unsafe_real_imag,
            constraint_type='unsafe',
            right_hand_value=right_hand_value
        )
        vmatr.append(Aub_unsafe)
        hmatr.append(bub_unsafe)

    # Generate dynamic constraints
    if dynamic_real_imag is not None:
        Aub_dynamic, bub_dynamic, Aeq_dynamic, beq_dynamic = generate_constraints(
            real_imag=dynamic_initial_real_imag,
            dynamic_real_imag=dynamic_real_imag,
            constraint_type='dynamic',
            right_hand_value=right_hand_value,
            eps=True
        )
        vmatr.append(Aub_dynamic)
        hmatr.append(bub_dynamic)

    if dynamic_real_k_imag is not None:
        Aub_k_dynamic, bub_k_dynamic, Aeq_k_dynamic, beq_k_dynamic = generate_constraints(
            real_imag=dynamic_initial_real_imag,
            dynamic_real_imag=dynamic_real_k_imag,
            constraint_type='dynamic',
            right_hand_value=0
        )
        vmatr.append(Aub_k_dynamic)
        hmatr.append(bub_k_dynamic)


    if constr:
        # Extract real and imaginary parts
        re = initial_real_imag[:, :, 0]
        im = initial_real_imag[:, :, 1]
        num_samples, num_terms = re.shape
        Aub_coeffs = np.zeros((1, 2 * num_terms), dtype=np.float64)

        # Append 1 for the epsilon variable
        Aub_eps = np.hstack([Aub_coeffs, k * np.ones((1, 1))])

        # -1 for y
        Aub_eps = np.hstack([Aub_eps, -1 * np.ones((1, 1))])


        # Append small value for < 0
        bub = [-1e-6]

        vmatr.append(Aub_eps)
        hmatr.append(bub)

    # Concatenate all inequality constraints
    Aub = np.vstack(vmatr)
    bub = np.hstack(hmatr)

    # Concatenate all equality constraints
    #Aeq = np.vstack([Aeq_initial, Aeq_unsafe, Aeq_dynamic])
    #beq = np.hstack([beq_initial, beq_unsafe, beq_dynamic])

    return Aub, bub, None, None


def construct_complex_coefficients(Re_a, Im_a):

    Re_a = np.array(Re_a)
    Im_a = np.array(Im_a)

    if Re_a.shape != Im_a.shape:
        raise ValueError("The real and imaginary parts must have the same shape.")

    a = Re_a + 1j * Im_a
    return a

def generate_barrier_polynomial(coeffs: list, terms:list, variables: list[sym.Symbol]) -> sym.Poly:
    p = [np.prod([t for t in term]) for term in terms]
    p = np.sum([coeffs[i] * p[i] for i in range(len(coeffs))])
    return sym.poly(p, variables, domain=sym.CC)

def solve_lp(c, Aub, bub, Aeq, beq, bounds, l, terms, vars, opt_meth):
    result = linprog(
        c,
        A_ub=Aub,
        b_ub=bub,
        A_eq=Aeq,
        b_eq=beq,
        bounds=bounds,
        method=opt_meth,
    )

    if result.success:
        x_optimal = result.x
        Re_a_optimal = x_optimal[0:2*l:2]
        Im_a_optimal = x_optimal[1:2 * l:2]
        y_optimal = x_optimal[-1]
        eps_optimal = x_optimal[-2]
        print("Y-OPTIMAL", y_optimal)
        print("EPS-OPTIMAL", eps_optimal)


        #print("Optimal y: ", y_optimal)
        # print("Real part a: ", Re_a_optimal)
        # print("Imaginary part a: ", Im_a_optimal)

        a_optimal = construct_complex_coefficients(Re_a_optimal, Im_a_optimal)
        #print("Complex coefficients a:", a_optimal)



        barrier_certificate = generate_barrier_polynomial(a_optimal, terms, vars)
        #print("Candidate Barrier Certificate: ", barrier_certificate)
        return barrier_certificate, a_optimal, y_optimal, eps_optimal

    else:
        print("Optimization failed:", result.message)
        return None, None, None, None


def print_linprog_problem(A_ub, b_ub, A_eq, b_eq, c):

    def format_coefficient(coeff):
        if coeff == 1:
            return ""
        elif coeff == -1:
            return "-"
        else:
            return f"{coeff}"

    n_vars = len(c)

    # Objective function
    objective_terms = []
    for i in range(n_vars):
        coeff = c[i]
        if coeff != 0:
            coeff_str = format_coefficient(coeff)
            term = f"{coeff_str}*x{i + 1}"
            objective_terms.append(term)
    objective_str = " + ".join(objective_terms)
    objective_str = objective_str.replace("+ -", "- ")
    print("Objective function:")
    print(f"Minimize: {objective_str}")
    print()

    # Inequality constraints
    if A_ub is not None and b_ub is not None:
        print("Inequality constraints:")
        for i in range(len(b_ub)):
            terms = []
            for j in range(n_vars):
                coeff = A_ub[i, j]
                if coeff != 0:
                    coeff_str = format_coefficient(coeff)
                    term = f"{coeff_str}*x{j + 1}"
                    terms.append(term)
            lhs = " + ".join(terms)
            lhs = lhs.replace("+ -", "- ")
            rhs = b_ub[i]
            print(f"{lhs} <= {rhs}")
        print()

    # Equality constraints
    if A_eq is not None and b_eq is not None:
        print("Equality constraints:")
        for i in range(len(b_eq)):
            terms = []
            for j in range(n_vars):
                coeff = A_eq[i, j]
                if coeff != 0:
                    coeff_str = format_coefficient(coeff)
                    term = f"{coeff_str}*x{j + 1}"
                    terms.append(term)
            lhs = " + ".join(terms)
            lhs = lhs.replace("+ -", "- ")
            rhs = b_eq[i]
            print(f"{lhs} == {rhs}")
        print()




