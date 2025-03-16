from src.verify_qc import *
from src.gates import *
from src.costants import *

if __name__ == '__main__':
    N_QUBIT = 1
    N_SAMPLES = 500
    DEG = 3

    Z = generate_symbols(N_QUBIT)

    Z0 = [
        {
            'variables': [Z[0]],
            'min': 0.701,
            'max': 1,
            'imConstr': {}
        }
    ]

    Zu = [
        {
            'variables': [Z[1]],
            'min': 0.601,
            'max': 1,
            'imConstr': {}
        }
    ]

    verifyQC(n_gate(Ygate,1), N_SAMPLES, Z, Z0, Zu, DEG, INTERIORPOINT, 1, 0.03)