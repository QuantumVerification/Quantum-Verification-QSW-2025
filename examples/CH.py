from src.verify_qc import *
from src.gates import *
from src.costants import *

if __name__ == '__main__':

    N_QUBIT = 2
    N_SAMPLES = 500
    DEG = 2


    Z = generate_symbols(N_QUBIT)
    print(Z)

    Z0 = [
        {
            'variables': [Z[0], Z[1]],
            'min': 0.8,
            'max': 1.0,
            'imConstr': {}
        }
    ]

    Zu = [
        {
            'variables': [Z[2], Z[3]],
            'min': 0.8,
            'max': 1,
            'imConstr': {}
        }
    ]

    verifyQC(CHgate, N_SAMPLES, Z, Z0, Zu, DEG, INTERIORPOINT, 4, 0.01)