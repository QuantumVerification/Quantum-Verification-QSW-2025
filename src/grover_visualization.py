'''
Input: N (il numero di elementi), M (il numero di soluzioni)
'''

import numpy as np
from scipy.optimize import linprog
from sympy.codegen import Print


def print_linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """
    Stampa in modo leggibile un problema di programmazione lineare per linprog di SciPy.

    Minimizza: c^T x
    Soggetto a: A_ub @ x <= b_ub (vincoli di disuguaglianza)
                A_eq @ x == b_eq (vincoli di uguaglianza)
                bounds (limiti sulle variabili)
    """
    print("Problema di Programmazione Lineare:")
    print("Minimizza:")
    obj_func = " + ".join([f"{c[i]:.2f}*x{i}" for i in range(len(c))])
    print(f"    {obj_func}")

    print("\nSoggetto a:")
    if A_ub is not None and b_ub is not None:
        print("  Vincoli di disuguaglianza:")
        for i in range(len(A_ub)):
            constraint = " + ".join([f"{A_ub[i][j]:.5f}*x{j}" for j in range(len(A_ub[i])) if A_ub[i][j] != 0])
            print(f"    {constraint} <= {b_ub[i]:.5f}")

    if A_eq is not None and b_eq is not None:
        print("  Vincoli di uguaglianza:")
        for i in range(len(A_eq)):
            constraint = " + ".join([f"{A_eq[i][j]:.2f}*x{j}" for j in range(len(A_eq[i])) if A_eq[i][j] != 0])
            print(f"    {constraint} == {b_eq[i]:.2f}")

    if bounds is not None:
        print("  Limiti sulle variabili:")
        for i, (lb, ub) in enumerate(bounds):
            print(f"    {lb if lb is not None else '-∞'} <= x{i} <= {ub if ub is not None else '∞'}")






# Parametri del problema
n = 8  # Numero di qubit
N = 2**n  # Numero totale di stati
M = N // 4 # Numero di soluzioni
#T = int(1/4 * np.pi * np.sqrt(N / M))  # Numero di passi
T = 2
NUM_SAMPLES = 30000
print(f'T = {T}')

# Campionamento degli stati
def sample_initial_states(M, N, perturbation=0.3):
    theta = np.arccos(np.sqrt((N - M) / N))
    print(f'theta = {theta}')
    return [np.arccos(np.sqrt((N - (M + perturbation * np.random.uniform(-1, 1))) / N))  for _ in range(NUM_SAMPLES)]


def sample_unsafe_states():
    return np.linspace(11/6 * np.pi, 15/8 * np.pi, NUM_SAMPLES)


def sample_system_states():
    return np.linspace(0, 2 * np.pi, NUM_SAMPLES)

initial_states = sample_initial_states(M, N)
unsafe_states = sample_unsafe_states()
system_states = sample_system_states()


# Dinamica del sistema
def system_dynamics(alpha, theta):
    return (2 * theta + alpha) % (2 * np.pi)

# Costruzione del problema di programmazione lineare
def build_linear_program(initial_states, unsafe_states, system_states, T):
    num_samples = len(initial_states) + len(unsafe_states) + len(system_states)
    c = [0, 0, 0, -1]  # Massimizzare gamma
    A = []
    b = []

    # Vincoli per gli stati iniziali
    for alpha in initial_states:
        A.append([alpha, -1, 0,0])
        b.append(0)

    # Vincoli per gli stati unsafe
    for alpha in unsafe_states:
        A.append([-alpha,0, 0, 1 ])
        b.append(0)

    # Vincoli per la dinamica del sistema
    for alpha in system_states:
        theta = np.arccos(np.sqrt((N - M) / N))

        alpha_next = system_dynamics(alpha, theta)
        # print(f'alpha = {alpha}, alpha_next = {alpha_next}, cos(G alpha) = {np.cos(alpha_next):.5f}, cos(alpha) = {np.cos(alpha):.5f}, B(Gz) - B(z) = {np.cos(alpha_next) - np.cos(alpha):.5f}')

        A.append([alpha_next - alpha, 0, -1, 0 ])
        b.append(0)

    # Vincolo aggiuntivo: gamma + delta * T < lambda
    A.append([0, 1, T, -1])
    b.append(-1e-5)



    return c, A, b

c, A, b = build_linear_program(initial_states, unsafe_states, system_states, T)
print(2 * np.pi)
# print_linprog(c, A, b, bounds=[(None, None),(None, None), (None, None), (0, 5)])
# Risoluzione del problema di programmazione lineare
result = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None),(None, None), (None, None), (0, 200)])

# Estrazione dei risultati
if result.success:

    c, gamma, delta, lambda_ = result.x
    print(f"c: {c}, gamma: {gamma}, delta: {delta}, lambda: {lambda_}")
else:
    print("Optimization failed:", result.message)