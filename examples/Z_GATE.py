from src.verify_qc import *
from src.gates import *
from src.costants import *

if __name__ == '__main__':
    N_QUBIT = 2
    N_SAMPLES = 3500
    DEG = 2

    Z = generate_symbols(N_QUBIT)

    Z0 = [
        {
            'variables': [Z[0]],
            'min': 0.9,
            'max': 1,
            'imConstr': {}
        }
    ]

    Zu = [
        {
            'variables': [Z[1]],
            'min': 0,
            'max': 0.1,
            'imConstr': {}
        }
    ]

    verifyQC(n_gate(Zgate,N_QUBIT), N_SAMPLES, Z, Z0, Zu, DEG, HIGHSIPM, 1)





