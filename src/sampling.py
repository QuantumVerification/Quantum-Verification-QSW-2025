import numpy as np
import sympy as sym
import random
from scipy.stats import qmc
import math

#from utils import *



def sample_states_qmc2(constraints, z, n_samples=100):
    """
    Sampling of quantum states using quasi-Monte Carlo methods (Sobol sequence).
    
    Parameters:
    - constraints: List of dictionaries, each with keys:
        - 'variables': list of sympy symbols (variables involved in the constraint)
        - 'min': minimum sum of probabilities over these variables
        - 'max': maximum sum of probabilities over these variables
        - 'imConstr': dictionary with keys as variables and values as (min_imag, max_imag)
    - z: List of sympy symbols representing the state variables
    - n_samples: Number of samples to generate
    
    Returns:
    - samples: List of dictionaries, each mapping variables to complex amplitudes
    """
    import numpy as np
    from scipy.stats import qmc
    import sympy as sym
    
    samples = []
    N = len(z)
    if len(constraints) == 0:
        constraints = [{'variables': [], 'max': 1.0, 'min': 0.0, 'imConstr': {}}]
    
    # Prepare a Sobol sampler for phases for all variables
    sampler_phases = qmc.Sobol(d=N, scramble=False)
    n_samples = 2 ** (math.ceil(math.log2(n_samples)))
    phase_samples = sampler_phases.random(n=n_samples)
    # Scale samples to [0, 2*pi]
    phase_samples = phase_samples * 2 * np.pi
    
    # Map variables to indices for phase sampling
    var_indices = {v: idx for idx, v in enumerate(z)}
    
    # Prepare Sobol samplers for S_constr and variable probabilities
    sampler_S = qmc.Sobol(d=1, scramble=False)
    S_constr_samples = sampler_S.random(n=n_samples).flatten()
    
    # Prepare Sobol sampler for variables in constraints
    max_d_vars = max(len(constr['variables']) for constr in constraints)
    sampler_vars = qmc.Sobol(d=max_d_vars - 1 if max_d_vars > 1 else 1, scramble=False)
    vars_samples = sampler_vars.random(n=n_samples)
    
    # Prepare Sobol sampler for remaining variables
    num_remaining_vars = N - max_d_vars
    if num_remaining_vars > 0:
        sampler_remain = qmc.Sobol(d=num_remaining_vars - 1 if num_remaining_vars > 1 else 1, scramble=False)
        remain_samples = sampler_remain.random(n=n_samples)
    else:
        remain_samples = None

    for idx in range(n_samples):
        sample = {v: 0 for v in z}
        assigned_vars = set()
        # Assign phases to all variables using the precomputed phase_samples
        phases = {}
        for v in z:
            var_idx = var_indices[v]
            ampl = phase_samples[idx, var_idx]
            phases[v] = ampl

        for constr in constraints:
            variables = constr['variables']
            min_sum = max(constr['min'], 0.0)
            max_sum = min(constr['max'], 1.0)
            if max_sum < min_sum:
                raise ValueError("Infeasible constraint: max_sum < min_sum")
            
            # Get S_constr for this sample
            S_constr = min_sum + S_constr_samples[idx] * (max_sum - min_sum)
            
            vars_to_assign = variables.copy()
            l = len(vars_to_assign)
            if l > 0:
                # Get the quasi-random numbers for the variables
                if l == 1:
                    random_points = []
                else:
                    random_points = vars_samples[idx, :l - 1].tolist()
                # Sort and scale the random points
                random_points = sorted(random_points)
                random_points = [0.0] + random_points + [1.0]
                # Compute the differences to get the proportions
                p_i_list = [S_constr * (random_points[i+1] - random_points[i]) for i in range(len(random_points) - 1)]
                # Assign probabilities and phases to variables
                for v, p_i in zip(vars_to_assign, p_i_list):
                    ampl = phases[v]  # Use the precomputed phase
                    zi = np.sqrt(p_i) * np.exp(1j * ampl)
                    if 'imConstr' in constr and v in constr['imConstr'].keys():
                        zi = normalizeComplex(zi, constr['imConstr'][v][0], constr['imConstr'][v][1])
                    sample[v] = zi
                    sample[sym.conjugate(v)] = np.conj(zi)
                    assigned_vars.add(v)
            else:
                S_constr = 0.0  # No probability mass assigned

        res = 1.0 - S_constr  # Residual probability mass

        # Assign remaining variables
        remaining_vars = [v for v in z if v not in assigned_vars]
        l_remain = len(remaining_vars)
        if l_remain > 0:
            # Get the quasi-random numbers for the remaining variables
            if l_remain == 1:
                random_points = []
            else:
                random_points = remain_samples[idx, :l_remain - 1].tolist()
            # Sort and scale the random points
            random_points = sorted(random_points)
            random_points = [0.0] + random_points + [1.0]
            # Compute the differences to get the proportions
            p_i_list = [res * (random_points[i+1] - random_points[i]) for i in range(len(random_points) - 1)]
            # Assign probabilities and phases to remaining variables
            for v, p_i in zip(remaining_vars, p_i_list):
                ampl = phases[v]  # Use the precomputed phase
                zi = np.sqrt(p_i) * np.exp(1j * ampl)
                sample[v] = zi
                sample[sym.conjugate(v)] = np.conj(zi)
        else:
            if abs(res) > 1e-8:
                raise ValueError("No variables left to assign, but residual probability is non-zero")
        
        # Append the sample
        samples.append(sample)
    
    return samples


def sample_states_qmc(constraints, z, n_samples=100):
    """
    Sampling of quantum states using quasi-Monte Carlo methods (Sobol sequence).
    
    Parameters:
    - constraints: List of dictionaries, each with keys:
        - 'variables': list of sympy symbols (variables involved in the constraint)
        - 'min': minimum sum of probabilities over these variables
        - 'max': maximum sum of probabilities over these variables
    - z: List of sympy symbols representing the state variables
    - n_samples: Number of samples to generate
    
    Returns:
    - samples: List of dictionaries, each mapping variables to complex amplitudes
    """
    samples = []
    N = len(z)
    if len(constraints) == 0:
        constraints = [{'variables': [], 'max': 1.0, 'min': 0.0}]
    
    for constr in constraints:
        variables = constr['variables']
        min_sum = max(constr['min'], 0.0)
        max_sum = min(constr['max'], 1.0)
        if max_sum < min_sum:
            raise ValueError("Infeasible constraint: max_sum < min_sum")
        
        n_samples= 2 ** (math.ceil(math.log2(n_samples)))
        # Generate quasi-random numbers for the total probability mass S_constr
        # Here, we use 1-dimensional Sobol sequence
        sampler_S = qmc.Sobol(d=1, scramble=False)
        S_constr_samples = sampler_S.random(n=n_samples).flatten()
        # Scale samples to [min_sum, max_sum]
        S_constr_samples = min_sum + S_constr_samples * (max_sum - min_sum)
        
        # Prepare the Sobol sampler for variable probabilities
        l = len(variables)
        sampler_vars = qmc.Sobol(d=l - 1 if l > 1 else 1, scramble=False)
        vars_samples = sampler_vars.random(n=n_samples)
        
        for idx in range(n_samples):
            S_constr = S_constr_samples[idx]
            sample = {v: 0 for v in z}
            res = 1.0 - S_constr  # Residual probability mass
            assigned_vars = set()
            vars_to_assign = variables.copy()
            l = len(vars_to_assign)
            if l > 0:
                # Get the quasi-random numbers for the variables
                if l == 1:
                    random_points = []
                else:
                    random_points = vars_samples[idx, :].tolist()
                # Sort and scale the random points
                random_points = sorted(random_points)
                random_points = [0.0] + random_points + [1.0]
                # Compute the differences to get the proportions
                p_i_list = [S_constr * (random_points[i+1] - random_points[i]) for i in range(len(random_points) - 1)]
                # Assign probabilities to variables
                for v, p_i in zip(vars_to_assign, p_i_list):
                    ampl = np.random.uniform(0, 2 * np.pi)
                    zi = np.sqrt(p_i) * np.exp(1j * ampl)
                    if v in constr['imConstr'].keys():
                        zi = normalizeComplex(zi, constr['imConstr'][v][0], constr['imConstr'][v][1])
                    sample[v] = zi
                    sample[sym.conjugate(v)] = np.conj(zi)
                    assigned_vars.add(v)
            else:
                res = 1.0  # All probability mass remains
            
            # Assign remaining variables
            remaining_vars = [v for v in z if v not in assigned_vars]
            l_remain = len(remaining_vars)
            if l_remain > 0:
                # Prepare a sampler for remaining variables
                sampler_remain = qmc.Sobol(d=l_remain - 1 if l_remain > 1 else 1, scramble=False)
                remain_samples = sampler_remain.random_base2(m=int(np.ceil(np.log2(n_samples))))
                remain_idx = idx % len(remain_samples)
                random_points = remain_samples[remain_idx, :].tolist() if l_remain > 1 else []
                # Sort and scale the random points
                random_points = sorted(random_points)
                random_points = [0.0] + random_points + [1.0]
                # Compute the differences to get the proportions
                p_i_list = [res * (random_points[i+1] - random_points[i]) for i in range(len(random_points) - 1)]
                # Assign probabilities to remaining variables
                for v, p_i in zip(remaining_vars, p_i_list):
                    ampl = np.random.uniform(0, 2 * np.pi)
                    zi = np.sqrt(p_i) * np.exp(1j * ampl)
                    sample[v] = zi
                    sample[sym.conjugate(v)] = np.conj(zi)
            else:
                if abs(res) > 1e-8:
                    raise ValueError("No variables left to assign, but residual probability is non-zero")
            
            # Append the sample
            samples.append(sample)
    
    return samples


def create_state_vectors(samples, Z):
    state_vectors = []
    for sample in samples:
        state = []
        for z in Z:
            '''
            state = []
            for e in sample.values():
                state.append(e)
            '''
            state.append(sample[z])
        state_vectors.append(state)
    return state_vectors

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
def separate_real_imag(sampled_terms) -> list[list[tuple]]:
    values = [[(value.real,value.imag) for value in sample] for sample in sampled_terms]
    return np.array(values)

'''
Get Re(a_i) and Im(a_i) as the values we are seeking
'''
def create_coefficients(p, coeff_real_tok='Re(a', coeff_imag_tok='Im(a'):
    real_coeff = [sym.Symbol(coeff_real_tok + str(i) + ')', complex=True) for i in range(len(p))]
    imag_coeff = [sym.Symbol(coeff_imag_tok + str(i) + ')', complex=True) for i in range(len(p))]
    return real_coeff, imag_coeff

def apply_op(sample_state, op):
    return np.dot(sample_state, op)


def generate_fx_samples(state_vectors, op, Z):
    dynamic_samples = []
    for state_vector in state_vectors:
        sample = {}
        new_state = apply_op(state_vector, op)
        for i, z in enumerate(Z):
            sample[z] = new_state[i]
            sample[z.conjugate()] = np.conj(new_state[i])
        dynamic_samples.append(sample)
    return dynamic_samples

def generate_k_fx_samples(state_vectors, ops, Z):
    dynamic_samples = []
    for state_vector in state_vectors:
        sample = {}
        new_state = state_vector
        for op in ops:
            new_state = apply_op(new_state, op)
        for i, z in enumerate(Z):
            sample[z] = new_state[i]
            sample[z.conjugate()] = np.conj(new_state[i])
        dynamic_samples.append(sample)
    return dynamic_samples

def normalizeComplex(z, imag_min, imag_max):
    # Estrai modulo
    r = np.abs(z)

    newImaginary = np.random.uniform(imag_min, imag_max)

    newReal = np.sqrt(r ** 2 - newImaginary ** 2)

    if np.real(z) < 0:
        newReal = -newReal

    newComplex = newReal + 1j * newImaginary

    return newComplex
