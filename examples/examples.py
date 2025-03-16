import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.utils import *
from src.find_BC import *
from src.grover_syn import *
from src.gates import *
from src.log import *
import argparse
import datetime
import logging


def Z_example(N_QUBIT):
    log_file = f"Zgate_{N_QUBIT}"
    
    Z = generate_symbols(N_QUBIT)
    var = generate_variables(Z)
    Z0 = [{'variables': [Z[0]], 'min': 0.9, 'max': 1.0, 'imConstr': {}}]
    Zu = [{'variables': [Z[0]], 'min': 0, 'max': 0.1, 'imConstr':{}}]
    unitaries = [n_gate(Zgate, N_QUBIT)]
    return Z, var, Z0, Zu, unitaries, log_file

def X_example(N_QUBIT):
    log_file = f"Xgate_{N_QUBIT}"
    
    Z = generate_symbols(N_QUBIT)
    var = generate_variables(Z)
    Z0 = [{'variables': [Z[0]], 'min': 0.9, 'max': 1.0, 'imConstr': {}}]
    Zu = [{'variables': [Z[0]], 'min': 0, 'max': 0.1, 'imConstr':{}}]
    unitaries = [n_gate(Xgate, N_QUBIT)]
    
    return Z, var, Z0, Zu, unitaries, log_file
    
def CX_example(N_QUBIT):
    log_file = f"CXgate_{N_QUBIT}"
    if N_QUBIT%2 != 0:
        raise Exception('Number of qubits should be even.')
    Z = generate_symbols(N_QUBIT)
    var = generate_variables(Z)
    Z0 = [{'variables': [Z[0]], 'min': 0.9, 'max': 1.0, 'imConstr': {}}]
    Zu = [{'variables': [Z[i] for i in range(len(Z)) if i != 0], 'min': 0.11, 'max': 1.0, 'imConstr':{}}]
    unitaries = [n_gate(CXgate, N_QUBIT//2)]
    
    return Z, var, Z0, Zu, unitaries, log_file 

def CZ_example(N_QUBIT):
    if N_QUBIT%2 != 0:
        raise Exception('Number of qubits should be even.')
    log_file = f"CZgate_{N_QUBIT}"
    Z = generate_symbols(N_QUBIT)
    var = generate_variables(Z)
    Z0 = [{'variables': [Z[0]], 'min': 0.9, 'max': 1.0, 'imConstr': {}}]
    Zu = [{'variables': [Z[0]], 'min': 0, 'max': 0.05, 'imConstr':{}}]
    unitaries = [n_gate(CZgate, N_QUBIT//2)]
    
    return Z, var, Z0, Zu, unitaries, log_file
    
def CX_CZ_example(N_QUBIT):
    if N_QUBIT%2 != 0:
        raise Exception('Number of qubits should be even.')
    log_file = f"CX_CZgate_{N_QUBIT}"
    Z = generate_symbols(N_QUBIT)
    var = generate_variables(Z)
    Z0 = [{'variables': [Z[0]], 'min': 0.9, 'max': 1.0, 'imConstr': {}}]
    Zu = [{'variables': [Z[0]], 'min': 0, 'max': 0.1, 'imConstr':{}}]
    unitaries = [n_gate(CXgate, N_QUBIT//2), n_gate(CZgate, N_QUBIT//2)]
    
    return Z, var, Z0, Zu, unitaries, log_file

def CXCZ_example(N_QUBIT):
    if N_QUBIT%2 != 0:
        raise Exception('Number of qubits should be even.')
    log_file = f"CXCZgate_{N_QUBIT}"
    Z = generate_symbols(N_QUBIT)
    var = generate_variables(Z)
    Z0 = [{'variables': [Z[0]], 'min': 0.9, 'max': 1.0, 'imConstr': {}}]
    Zu = [{'variables': [Z[0]], 'min': 0, 'max': 0.1, 'imConstr':{}}]
    unitaries = [np.dot(n_gate(CXgate, N_QUBIT//2), n_gate(CZgate, N_QUBIT//2))]

    return

def H_example(N_QUBIT):
    log_file = f"Hgate_{N_QUBIT}"
    Z = generate_symbols(N_QUBIT)
    var = generate_variables(Z)
    Z0 = [{'variables': [Z[0]], 'min': 0.9, 'max': 1.0, 'imConstr': {}}]
    Zu = [{'variables': [Z[0]], 'min': 0, 'max': 0.1, 'imConstr':{}}]
    unitaries = [n_gate(Hgate, N_QUBIT)]

    return Z, var, Z0, Zu, unitaries, log_file
# SWAP todo
def SWAP_example(N_QUBIT):
    log_file = f"SWAPgate_{N_QUBIT}"
    Z = generate_symbols(N_QUBIT)
    var = generate_variables(Z)
    Z0 = [{'variables': [Z[1]], 'min': 0.9, 'max': 1.0, 'imConstr': {}}]
    Zu = [{'variables': [Z[0]], 'min': 0.5, 'max': 1.0, 'imConstr':{}}]
    unitaries = [n_gate(Sgate, N_QUBIT)]

    return Z, var, Z0, Zu, unitaries, log_file

def T_example(N_QUBIT):
    log_file = f"Tgate_{N_QUBIT}"
    Z = generate_symbols(N_QUBIT)
    var = generate_variables(Z)
    Z0 = [{'variables': [Z[0]], 'min': 0.9, 'max': 1.0, 'imConstr': {}}]
    Zu = [{'variables': [Z[0]], 'min': 0, 'max': 0.1, 'imConstr':{}}]
    unitaries = [n_gate(Tgate, N_QUBIT)]
    return Z, var, Z0, Zu, unitaries, log_file

def Grover_example(N_QUBIT):
    log_file = f"GROVER_simple_{N_QUBIT}"
    Z = generate_symbols(N_QUBIT)
    var = generate_variables(Z)
    Z0 = [{'variables': [Z[i]], 'min': 1 / (2 ** N_QUBIT), 'max': 1 / (2 ** N_QUBIT),
       'imConstr': {Z[i]: (-np.sqrt(1 / (10 ** (N_QUBIT+1))) , np.sqrt(1 / (10 ** (N_QUBIT+1))))}} for i in range(2 ** N_QUBIT)]
    Zu = [{'variables': [Z[0]], 'min': 0.9, 'max': 1.0, 'imConstr':{}}]
    N = 2**N_QUBIT
    mark = 1
    oracle = np.eye(N, N)
    oracle[mark, mark] = -1
    diffusion_oracle = np.eye(N, N)
    temp = np.zeros((N, N))
    temp[0, 0] = 1
    diffusion_oracle = 2 * temp - diffusion_oracle
    diffusion = np.dot(n_gate(Hgate, N_QUBIT), np.dot(diffusion_oracle, n_gate(Hgate, N_QUBIT)))
    diffusion = np.round(diffusion, decimals=10)
    G = np.dot(diffusion, oracle)
    unitaries = [G]
    return Z, var, Z0, Zu, unitaries, log_file

def run_example(Z, var, N_SAMPLES, log_file, Z0, Zu, unitaries, poly_degree=2, k=1, epsilon=0.01, gamma=0.01, smt_timeout=300):
    logger = set_logger(log_file + ".log")
    logger.info(str(datetime.datetime.now()))
    logger.info(f"Storing logs in {logger.handlers[-1].baseFilename}")
    logger.info(f"Running {log_file} example")
    find_barrier(Z, var, unitaries, N_SAMPLES, Z0, Zu, gamma, epsilon, k, poly_degree)
    return

parser = argparse.ArgumentParser(
    description="Run an example.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
examples = ['xgate','zgate', 'cx_czgate', 'cxczgate','cx','cnot','cz', 'swap', 'tgate','grover_simple', 'grover_geo']
parser.add_argument("--example", "-ex", type=str, choices=examples, required=True, help="Example to run.")
#parser.add_argument("--runs", type=int, default=1, help="Number of runs to perform.")
parser.add_argument("-n", type=int, default=1, help="Number of qubits.")
parser.add_argument("-samples", type=int, default=1, help="Number of samples.")
parser.add_argument("--epsilon", "-eps", type=float, default=0.01, metavar="EPS", help="Bound for difference condition (B_t(f_t(x)) - B_t(x) < EPS).")
parser.add_argument("--gamma", "-gam", type=float, default=0.01, metavar="GAM", help="Bound for change condition (B_{t+1}(x) - B_t(x) < GAM).")
parser.add_argument("-k", type=int, default=1, help="Inductive parameter.")
parser.add_argument("--barrier-degree", type=int, default=2, help="Maximum degree of generated barrier.")
parser.add_argument("--smt-timeout", type=int, default=300, help="Set timeout for SMT solvers.")
parser.add_argument("--err", type=float, default=0.5, help="Error margin to perturbate M yielding uncertainties.")
parser.add_argument("--mu", type=float, default=0.3, help="Deviation bound on the grover operator G, to capture noisy dynamic.")


if __name__ == '__main__':
    args = parser.parse_args()
    if args.example == 'zgate':
        Z, var, Z0, Zu, unitaries, log_file = Z_example(N_QUBIT=args.n)
    elif args.example == 'xgate':
        Z, var, Z0, Zu, unitaries, log_file = X_example(N_QUBIT=args.n)
    elif args.example == 'cx_cz_gate':
        Z, var, Z0, Zu, unitaries, log_file = CX_CZ_example(N_QUBIT=args.n)
    elif args.example == 'cxcz_gate':
        Z, var, Z0, Zu, unitaries, log_file = CXCZ_example(N_QUBIT=args.n)
    elif args.example == 'cnot' or args.example == 'cx':
        Z, var, Z0, Zu, unitaries, log_file = CX_example(N_QUBIT=args.n)
    elif args.example == 'cz':
        Z, var, Z0, Zu, unitaries, log_file = CZ_example(N_QUBIT=args.n)
    elif args.example == 'hgate':
        Z, var, Z0, Zu, unitaries, log_file = H_example(N_QUBIT=args.n)
    elif args.example == 'tgate':
        Z, var, Z0, Zu, unitaries, log_file = T_example(N_QUBIT=args.n)
    elif args.example == 'swap':
        Z, var, Z0, Zu, unitaries, log_file = SWAP_example(N_QUBIT=args.n,)
    elif args.example == 'grover_simple':
        Z, var, Z0, Zu, unitaries, log_file = Grover_example(N_QUBIT=args.n)
    elif args.example == 'grover_geo':
        logger = set_logger(f"Grover_geo_{args.n}" + ".log")
        logger.info(str(datetime.datetime.now()))
        logger.info(f"Storing logs in {logger.handlers[-1].baseFilename}")
        logger.info(f"Running grover's geometric visualization example")
        # find_BC_grover(N_QUBIT=args.n, n_samples=args.samples, err=args.err, mu=args.mu)
    else: raise Exception(f"Invalid example. Choose one from {examples}")
    
    if args.example == 'grover_geo': find_BC_grover(N_QUBIT=args.n, n_samples=args.samples, err=args.err, mu=args.mu)
    else: run_example(Z, var, args.samples, log_file, Z0, Zu, unitaries, poly_degree=args.barrier_degree, k=args.k, epsilon=args.epsilon, gamma=args.gamma, smt_timeout=args.smt_timeout)