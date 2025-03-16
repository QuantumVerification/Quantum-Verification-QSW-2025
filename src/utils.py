import numpy as np
import sympy as sym
import itertools
from typing import List, Tuple
from sympy import I
import time

'''
Given the number of the qubits, generate the symbols
'''
def generate_symbols(n : int) -> list[sym.Symbol]:
    return [sym.Symbol('z' + str(i), complex=True) for i in range(2**n)]

def generate_variables(Z: list[sym.Symbol]) -> list[sym.Symbol]:
    variables = Z + [z.conjugate() for z in Z]
    return variables

'''
Given the variables we generate all possible terms for the polynomial
'''
def generate_terms(variables : list[sym.Symbol], deg=2) -> list[tuple]:
    p = []
    for d in range(deg + 1):
        p += itertools.combinations_with_replacement(variables, d) # get all possible combinations of variables
    return p

def construct_complex_coefficients(Re_a, Im_a):

    Re_a = np.array(Re_a)
    Im_a = np.array(Im_a)

    if Re_a.shape != Im_a.shape:
        raise ValueError("The real and imaginary parts must have the same shape.")

    a = Re_a + 1j * Im_a
    return a

def generate_barrier_polynomial(coeffs: list, terms:list, variables: list[sym.Symbol]) -> sym.Poly:
    p = [np.prod([t for t in term]) for term in terms]
    p = np.sum([coeffs[i] * p[i] for i in range(len(coeffs))])
    return sym.poly(p, variables, domain=sym.CC)


def round_values(values, precision_bound = 1e-10):
    n_digits = -int(np.log10(precision_bound))
    rounded_values = [round(value, n_digits) if abs(value) > precision_bound else 0 for value in values]
    return rounded_values





