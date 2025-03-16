import random
import numpy as np
import sympy as sym
from scipy.stats import qmc


def sample_states5(constraints, z, n_samples=100):
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
                p_i_list = [S_constr * (random_points[i + 1] - random_points[i]) for i in range(len(random_points) - 1)]
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
            p_i_list = [res * (random_points[i + 1] - random_points[i]) for i in range(len(random_points) - 1)]
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


'''
    Sampling of quantum states with uniform distribution.
'''
def sample_states2(constr: dict, z: list , n_samples = 100):
    samples = []
    N = len(z)

    keys = [j for j in constr.keys()]
    # Verify constraints
    if len(constr) > N:
        raise ValueError("Numeri di condizioni superiori al limite")

    # Genera campioni
    for i in range(n_samples):
        sample = {v:0 for v in z}
        constrained_probs = {}

        res = 1.0

        random.shuffle(keys)
        not_constr = [k for k in z if k not in keys]
        l = len(not_constr)
        for j in keys:
            rq = np.random.uniform(constr[j][0], min(constr[j][1], res))
            res -= rq
            ampl = np.random.uniform(0, 2 * np.pi)
            zi = np.sqrt(rq) * np.exp(1j*ampl)
            sample[j] = zi
            sample[sym.conjugate(j)] = np.conj(zi)
        if res < 0:
            raise ValueError("Probabilità residua inferiore a 0")

        if not_constr:
            dirichlet_params = np.ones(len(not_constr))
            unconstrained_probs = np.random.dirichlet(dirichlet_params) * res
        else:
            unconstrained_probs = []
        for var, prob in zip(not_constr, unconstrained_probs):
            amplitude = np.sqrt(prob) * np.exp(1j * np.random.uniform(0, 2 * np.pi))
            sample[var] = amplitude
            sample[sym.conjugate(var)] = np.conj(amplitude)
        samples.append(sample)
    return samples

'''
def check_samples(samples, constr):
    for i in range(len(samples)):
        sum = 0.0
        #print(f"---SAMPLE NUMBER {i}---")
        constr_keys = [k for k in constr.keys()]
        for key, val in samples[i].items():
            modq = np.abs(val) ** 2
            sum += modq
            if key in constr_keys:
                if constr[key][0] <= modq <= constr[key][1]:
                    #print(f'|{val}|^2 = {modq} è compreso tra {constr[key][0]} e {constr[key][1]}')
                    print()
                else:
                    raise ValueError("Errore nei vincoli!")
            else:
                #print(f'|{val}|^2 = {modq}')
                print()
        #if not(1.99 <= sum <= 2.01):
        #    raise ValueError("Somma != 1...")
        # print(f'Total sum = {sum}')ß
        print()
'''
def check_samples(samples, z):
    for i in range(len(samples)):
        sum = 0
        #print(f"---SAMPLE NUMBER {i}---")
        for j in samples[i].values():
            modq = np.abs(j) ** 2
            sum += modq
            print(f'|{j}|^2 = {modq}')
        print(f'Total sum = {sum}')
        print()
def create_state_vectors(samples, Z):
    state_vectors = []
    for sample in samples:
        #print(sample)
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


def normalizeComplex(z, imag_min, imag_max):
    # Estrai modulo
    r = np.abs(z)

    newImaginary = np.random.uniform(imag_min, imag_max)

    newReal = np.sqrt(r ** 2 - newImaginary ** 2)

    if np.real(z) < 0:
        newReal = -newReal

    newComplex = newReal + 1j * newImaginary

    return newComplex

def sample_states(constraints, z, n_samples=100):
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
                p_i_list = [S_constr * (random_points[i + 1] - random_points[i]) for i in range(len(random_points) - 1)]
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
                p_i_list = [res * (random_points[i + 1] - random_points[i]) for i in range(len(random_points) - 1)]
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

def sample_states2(constraints, z, n_samples=100):
    """
    Sampling of quantum states with uniform distribution over the feasible region defined by sum constraints.

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

    for _ in range(n_samples):
        sample = {v: 0 for v in z}
        '''
        for k in z:
            sample[sym.conjugate(k)] = 0
        '''
        res = 1  # Residual probability mass
        assigned_vars = set()
        # Process sum constraints
        for constr in constraints:
            variables = constr['variables']
            # Adjust min and max sums within residual probability
            min_sum = max(constr['min'], 0.0)
            max_sum = min(constr['max'], res)


            if max_sum < min_sum:
                raise ValueError("Infeasible constraint: max_sum < min_sum")

            # Sample total probability mass for this constraint uniformly
            S_constr = np.random.uniform(min_sum, max_sum)
            res -= S_constr
            if res < 0:
                raise ValueError("Residual probability less than 0 after processing constraint")

            # Variables not yet assigned in this constraint
            vars_to_assign = [v for v in variables if v not in assigned_vars]
            l = len(vars_to_assign)
            if l == 0:
                continue  # All variables already assigned

            # Shuffle variables for randomness
            random.shuffle(vars_to_assign)
            # Generate l - 1 sorted random numbers between 0 and S_constr
            random_points = sorted(np.random.uniform(0, S_constr, l - 1))
            breaks = [0.0] + random_points + [S_constr]
            p_i_list = [breaks[i + 1] - breaks[i] for i in range(l)]

            # Assign probabilities to variables
            for v, p_i in zip(vars_to_assign, p_i_list):
                ampl = np.random.uniform(0, 2 * np.pi)
                zi = np.sqrt(p_i) * np.exp(1j * ampl)

                if v in constr['imConstr'].keys():
                    zi = normalizeComplex(zi, constr['imConstr'][v][0], constr['imConstr'][v][1])


                sample[v] = zi
                sample[sym.conjugate(v)] = np.conj(zi)
                assigned_vars.add(v)

        # Assign remaining variables
        remaining_vars = [v for v in z if v not in assigned_vars]
        l = len(remaining_vars)
        if l > 0:
            if res < -1e-8:
                raise ValueError("Residual probability less than 0 when assigning remaining variables")
            elif res > 1e-8:
                # Shuffle variables for randomness
                random.shuffle(remaining_vars)
                # Generate l - 1 sorted random numbers between 0 and res
                random_points = sorted(np.random.uniform(0, res, l - 1))
                breaks = [0.0] + random_points + [res]
                p_i_list = [breaks[i + 1] - breaks[i] for i in range(l)]

                # Assign probabilities to variables
                for v, p_i in zip(remaining_vars, p_i_list):
                    ampl = np.random.uniform(0, 2 * np.pi)
                    zi = np.sqrt(p_i) * np.exp(1j * ampl)
                    sample[v] = zi
                    sample[sym.conjugate(v)] = np.conj(zi)
                    assigned_vars.add(v)
                res = 0.0
            else:
                # No residual probability left, assign zero to remaining variables
                for v in remaining_vars:
                    sample[v] = 0.0
                    sample[sym.conjugate(v)] = 0.0
        '''       
        else:
            if res > 1e-8:
                print("res: ", res)
                #raise ValueError("No variables left to assign, but residual probability is non-zero") '''
        samples.append(sample)
    return samples



