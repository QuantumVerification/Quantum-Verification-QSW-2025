from src.verify_qc import *
from src.gates import *
from src.costants import *

if __name__ == '__main__':

    N_QUBIT = 2
    N_SAMPLES = 8000
    DEG = 2


    Z = generate_symbols(N_QUBIT)
    print(Z)

    Z0 = [
        {
            'variables': [Z[0]],
            'min': 0.9,
            'max': 1.0,
            'imConstr': {}
        }
    ]

    Zu = [
        {
            'variables': [Z[1],Z[2],Z[3]],
            'min': 0.11,
            'max': 1,
            'imConstr': {}
        }
    ]
    print(Zu)
    Zu = [
        {
            'variables': [Z[i] for i in range(N_QUBIT)],
            'min': 0.11,
            'max': 1,
            'imConstr': {}
        }
    ]

    verifyQC(n_gate(CXgate, N_QUBIT // 2), N_SAMPLES, Z, Z0, Zu, DEG, HIGHS, 1)