import numpy as np

from src.costants import HIGHSIPM, INTERIORPOINT
from src.verify_qc import *
from src.gates import *

if __name__ == "__main__":

    N_QUBIT = 5
    N_SAMPLES = 200
    DEG = 4

    Z = generate_symbols(N_QUBIT)
    print(Z)

    N = 2**N_QUBIT
    mark = 1

    oracle = np.eye(N, N)
    oracle[mark, mark] = -1
    diffusion_oracle = np.eye(N, N)
    temp = np.zeros((N, N))
    temp[0, 0] = 1
    diffusion_oracle = 2 * temp - diffusion_oracle

    diffusion = np.dot(n_gate(Hgate, N_QUBIT), np.dot(diffusion_oracle, n_gate(Hgate, N_QUBIT)))
    circuit = np.dot(diffusion, oracle)
    print(circuit)

    Z0 = [
        {
            'variables': [Z[i]],
            'min': 1 / (2**N_QUBIT)  ,
            'max': 1 / (2**N_QUBIT) ,
            'imConstr': {Z[i] : (-np.sqrt(1 / (10**(N_QUBIT + 1))), np.sqrt(1 / (10**(N_QUBIT + 1))))}} for i in range(2**N_QUBIT)
    ]



    Zu = [
        {
            'variables': [Z[3]],
            'min': 0.9,
            'max': 1,
            'imConstr' : {}
        }
    ]


    verifyQC(circuit, N_SAMPLES, Z, Z0, Zu, DEG, HIGHSIPM, 3)