import numpy as np
from scipy.constants import precision
from sympy import false, Poly, conjugate
from fractions import Fraction

from sympy.abc import epsilon

from src.utils import *
from src.gates import *
from z3.z3 import *
import sympy as sym
from sympy.core import *
from colorama import Fore, Style



def _sympy_to_z3_rec(var_map, e):
    'recursive call for sympy_to_z3()'

    rv = None

    if not isinstance(e, Expr):
        raise RuntimeError("Expected sympy Expr: " + repr(e))

    if isinstance(e, sym.re) or isinstance(e, sym.im):
        rv = var_map.get(e)

        if rv == None:
            raise RuntimeError("No var was corresponds to symbol '" + str(e) + "'")

    elif isinstance(e, Number):
        rv = float(e)
    elif isinstance(e, Mul):
        rv = _sympy_to_z3_rec(var_map, e.args[0])

        for child in e.args[1:]:
            rv *= _sympy_to_z3_rec(var_map, child)
    elif isinstance(e, Add):
        rv = _sympy_to_z3_rec(var_map, e.args[0])

        for child in e.args[1:]:
            rv += _sympy_to_z3_rec(var_map, child)
    elif isinstance(e, Pow):
        term = _sympy_to_z3_rec(var_map, e.args[0])
        exponent = _sympy_to_z3_rec(var_map, e.args[1])

        if exponent == 0.5:
            # sqrt
            rv = Sqrt(term)
        else:
            rv = term**exponent

    if rv == None:
        raise RuntimeError("Type '" + str(type(e)) + "' is not yet implemented for convertion to a z3 expresion. " + \
                            "Subexpression was '" + str(e) + "'.")

    return rv

def genZ3Constr(Z, var_z3_dict):
    res = []
    for i in range(len(Z)):
        sum = []
        for v in Z[i]['variables']:
            sum.append(var_z3_dict[sym.re(v)]**2)
            sum.append(var_z3_dict[sym.im(v)]**2)
            if v in Z[i]['imConstr']:
                res.append(var_z3_dict[sym.im(v)] >= Z[i]['imConstr'][0])
                res.append(var_z3_dict[sym.im(v)] <= Z[i]['imConstr'][1])


        res.append(Sum(sum) >= Z[i]['min'])
        res.append(Sum(sum) <= Z[i]['max'])


    return res



def check_barrier(Z:list[sym.Symbol],
                  barrier:sym.Poly,
                  unitary,
                  var,
                  Z0,
                  Zu,
                  K,
                  epsilon,
                  delta,
                  timeout=300):

    Z_RI = [sym.re(z) for z in Z] + [sym.im(z) for z in Z]
    FZ = np.dot(unitary, Z)

    mapping = {}
    for i in range(len(Z)):
        mapping[Z[i]] = FZ[i]
        mapping[conjugate(Z[i])] = conjugate(FZ[i])


    real_barrier = sym.poly(sym.re(sym.expand_complex(barrier.as_expr())))
    print("REAL BARRIER: ", real_barrier)
    #print("F: ", sym.poly(barrier.as_expr().xreplace(mapping)))
    #fp = sym.poly(barrier.as_expr().xreplace(mapping) - barrier,var)
    #print("SUBS: ",barrier.subs(mapping, simultaneous=True))



    #fp = sym.poly(barrier.subs(mapping, simultaneous=True) - barrier, var)
    fp = sym.poly(barrier.as_expr().subs(zip(Z, np.dot(unitary, Z)), simultaneous=True) - barrier, var)
    current_expr = barrier.as_expr()
    for _ in range(K):
        # Aggiorna l'espressione applicando l'unitary
        current_expr = current_expr.subs(zip(Z, np.dot(unitary, Z)), simultaneous=True)

    fp_k = sym.poly(current_expr - barrier, var)
    fp_k = sym.re((sym.expand_complex(fp_k.as_expr())))
    fp = sym.re((sym.expand_complex(fp.as_expr())))
    #print("FPreal: ", fp)

    
    
    var_z3_dict = dict(zip(Z_RI, [Real(str(var)) for var in Z_RI]))
    k = [v for v in var_z3_dict.values()]
    norm = Sum([var**2 for var in k]) == 1


    
    z3_barrier = _sympy_to_z3_rec(var_z3_dict, real_barrier.as_expr())
    #z3_barrier = round_expr_coeffs(z3_barrier, 3)

    z3_fx_barrier = _sympy_to_z3_rec(var_z3_dict, fp)
    z3_fx_k_barrier = _sympy_to_z3_rec(var_z3_dict, fp_k)
    #print("Z3_FX_BARRIER: ", z3_fx_barrier)
    #print("Z3_FKX_BARRIER: ",z3_fx_k_barrier)
    #print(z3_barrier)



    #checkZ0 = genZ3Constr(Z0, var_z3_dict)

    initial_conditions = [norm, z3_barrier > 1e-6]


    for constr in Z0:
        variables = constr['variables']
        min_sum = constr['min']
        max_sum = constr['max']
        # Sum over the modulus squared of the variables
        sum_modulus_squared = sum(
            var_z3_dict[sym.re(v)]**2 + var_z3_dict[sym.im(v)]**2 for v in variables
        )
        # Add the inequality constraints
        if min_sum is not None:
            initial_conditions.append(sum_modulus_squared >= min_sum)
        if max_sum is not None:
            initial_conditions.append(sum_modulus_squared <= max_sum)
        for k, v in constr['imConstr'].items():
            initial_conditions.append(var_z3_dict[sym.im(k)] >= v[0])
            initial_conditions.append(var_z3_dict[sym.im(k)] <= v[1])

    
    print(initial_conditions)
    i_sample = run_check(initial_conditions, 0)
    init_add_sample = build_dictionary(i_sample, Z)

    if init_add_sample != {}:
        return [init_add_sample], [{}], [{}]



    #checkZU = genZ3Constr(Zu, var_z3_dict
    print(delta)
    unsafe_conditions = [norm, z3_barrier <= delta - 1e-3]


    for constr in Zu:
        variables = constr['variables']
        min_sum = constr['min']
        max_sum = constr['max']
        # Sum over the modulus squared of the variables
        sum_modulus_squared = sum(
            var_z3_dict[sym.re(v)]**2 + var_z3_dict[sym.im(v)]**2 for v in variables
        )
        # Add the inequality constraints
        if min_sum is not None:
            unsafe_conditions.append(sum_modulus_squared >= min_sum)
        if max_sum is not None:
            unsafe_conditions.append(sum_modulus_squared <= max_sum)
        for k, v in constr['imConstr'].items():
            unsafe_conditions.append(var_z3_dict[sym.im(k)] >= v[0])
            unsafe_conditions.append(var_z3_dict[sym.im(k)] <= v[1])




    u_sample = run_check(unsafe_conditions, 1)
    print(unsafe_conditions)
    unsafe_add_sample = build_dictionary(u_sample, Z)
    if unsafe_add_sample != {}:
        return [{}], [unsafe_add_sample], [{}]

    if (z3_fx_barrier > epsilon) == False and (z3_fx_k_barrier > epsilon) == False:
        print(f"Nessuna soluzione trovata: {z3_fx_barrier}, {z3_fx_k_barrier}")
        return [init_add_sample], [unsafe_add_sample], [{}]

    dynamic_constraint = [norm, z3_fx_barrier > epsilon + 1e-8]

    delta_sample = run_check(dynamic_constraint, 2)

    delta_add_sample = build_dictionary(delta_sample, Z)
    if delta_add_sample != {}:
        return [{}], [{}], [delta_add_sample]


    dynamic_constraint = [norm, z3_fx_k_barrier > 1e-6]

    delta_sample = run_check(dynamic_constraint, 3)

    delta_add_sample = build_dictionary(delta_sample, Z)
    if delta_add_sample != {}:
        return [{}], [{}], [delta_add_sample]



    return [init_add_sample], [unsafe_add_sample], [delta_add_sample]

def run_check(conditions, n):
    #tactic = Then('solve-eqs','smt')
    #set_option(timeout=1000)

    s = SolverFor('QF_NRA')
    #s = Solver()
    #s = tactic.solver()
    #s.set(timeout=61900)


    [s.add(cond) for cond in conditions]
    #print(s.to_smt2())
    #print(toSMT2Benchmark(s))
    if s.check() == sat:
        m = s.model()
        variables = {}
        for decl in m.decls():
            var_name = decl.name()
            variables[var_name] = z3_value_to_float(m[decl])
        print(Fore.RED + f'Contro esempio -------------------> {variables}')
        return variables
    else:
        print(Fore.YELLOW + "Nessuna soluzione trovata.")
        file_path = f'./smtfiles/output{n}.smt2'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            f.write(s.to_smt2())




    return {}


# Converte valori di z3 in float
def z3_value_to_float(value):
    if isinstance(value, RatNumRef):
        return float(value.numerator_as_long()) / value.denominator_as_long()
    else:
        return float(value.as_decimal(10).rstrip('?'))


def build_dictionary(sample, Z):
    if sample != {}:
        unsafe_add_sample = {v: 0 for v in Z}
        for i in range(len(unsafe_add_sample)):
            cn = np.complex128(sample["re(z" + str(i) + ")"] + sample["im(z" + str(i) + ")"] * 1j)
            unsafe_add_sample[Z[i]] = cn
            unsafe_add_sample[sym.conjugate(Z[i])] = np.conjugate(cn)
        return unsafe_add_sample
    return {}
