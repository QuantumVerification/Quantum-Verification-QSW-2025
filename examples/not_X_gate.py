from src.verify_qc import *
from src.gates import *
from src.costants import *

if __name__ == '__main__':
    N_QUBIT = 3
    N_SAMPLES = 10000
    DEG = 4
    Z = generate_symbols(N_QUBIT)
    print(Z)




    Z0 = [{'variables': [Z[i]],
           'min': 1/(2**N_QUBIT) ,
           'max': 1/(2**N_QUBIT) ,
           'imConstr': {}} for i in range(2**N_QUBIT)]

    Zu = [
        {
            'variables': [Z[1]],
            'min': 0.8,
            'max': 1,
            'imConstr': {}
        }
    ]

    verifyQC(n_gate(Xgate, N_QUBIT), N_SAMPLES, Z, Z0, Zu, DEG, HIGHSIPM,1)
