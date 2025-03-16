from src.verify_qc import *
from src.gates import *
from src.costants import *

if __name__ == '__main__':
    N_QUBIT = 1
    N_SAMPLES = 4000
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
            'variables': [Z[0]],
            'min': 0,
            'max': 0.1,
            'imConstr': {}
        }
    ]

    verifyQC(Hgate, N_SAMPLES, Z, Z0, Zu, DEG, HIGHSIPM, 40)

