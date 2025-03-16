import z3
import sympy as sym
import numpy as np
from sympy.core import *
from z3.z3 import *
from src.log import *
import logging

logger = logging.getLogger("smtcheck")

def expr_to_smtlib(expr):
    """
    Recursively convert a Sympy expression to an SMT-LIB string in prefix notation.
    This function handles basic operations: addition, multiplication, and power expressions,
    where integer exponents are expanded using repeated multiplication (and division for negative exponents).
    """
    # If the expression is a number, return its string representation.
    if expr.is_Number:
        return str(expr)
    
    # If the expression is a symbol, assume its name is already in the SMT form.
    if expr.is_Symbol:
        return str(expr)
    
    # Handle power expressions.
    if expr.is_Pow:
        base, exponent = expr.as_base_exp()
        # If exponent is an integer, expand it.
        if exponent.is_Integer:
            exp_val = int(exponent)
            base_str = expr_to_smtlib(base)
            if exp_val == 0:
                # x^0 = 1
                return "1"
            elif exp_val == 1:
                return base_str
            else:
                # Multiply base string exp_val times.
                return f"(* {' '.join([base_str] * exp_val)})"
        else:
            # For non-integer exponents, fallback to an explicit multiplication
            # if possible, or raise an error.
            raise NotImplementedError("Non-integer exponents are not supported in SMT-LIB conversion.")
    
    # Handle multiplication.
    if expr.is_Mul:
        args_smt = [expr_to_smtlib(arg) for arg in expr.args]
        return f"(* {' '.join(args_smt)})"
    
    # Handle addition.
    if expr.is_Add:
        args_smt = [expr_to_smtlib(arg) for arg in expr.args]
        return f"(+ {' '.join(args_smt)})"
    
    # Fallback: use the default string conversion.
    return str(expr)


def bitwise_dot(i, j, N):
    """Compute the bitwise dot product mod 2 of i and j assuming N-bit numbers."""
    dot = 0
    for k in range(N):
        dot += ((i >> k) & 1) * ((j >> k) & 1)
    return dot % 2

def barrier_to_smt_h(B_expr, N):
    """
    Given a Sympy expression B_expr (the real part of the barrier certificate)
    defined in terms of variables z_0_re, z_0_im, ..., z_{M-1}_re, z_{M-1}_im (with M = 2**N)
    and an integer N (number of qubits), return an SMT-LIB string encoding:
    
      1. Declaration of 2**N complex amplitudes (separated into real and imaginary parts).
      2. Normalization: sum_{i=0}^{2**N -1} (z_i_re^2 + z_i_im^2) = 1.
      3. Declaration of constant s for √2: s > 0 and s*s = 2.
      4. Definition of B_original as a function of the original amplitudes.
      5. Definition of the H gate transformation:
             For each j, define
             (z'_j)_re = (1/(s**N)) * ∑_{i=0}^{M-1} (-1)^(dot(i,j)) * z_i_re,
             (z'_j)_im = (1/(s**N)) * ∑_{i=0}^{M-1} (-1)^(dot(i,j)) * z_i_im.
      6. Definition of B_transformed = B_original evaluated on the transformed amplitudes.
      7. Universal assertion that for all original state values, B_transformed - B_original <= 0.
      
    Returns an SMT-LIB string.
    """
    M = 2**N
    smt_lines = []
    
    # Set logic (we need NRA because of non-linear arithmetic and quantifiers)
    smt_lines.append("(set-logic QF_NRA)")
    
    # Declare original state variables: z_0_re, z_0_im, ..., z_{M-1}_re, z_{M-1}_im
    for i in range(M):
        smt_lines.append(f"(declare-fun rez{i} () Real)")
        smt_lines.append(f"(declare-fun imz{i} () Real)")
    
    # Normalization condition: sum_{i=0}^{M-1} (z_i_re^2 + z_i_im^2) = 1
    norm_terms = []
    for i in range(M):
        norm_terms.append(f"(* rez{i} rez{i})")
        norm_terms.append(f"(* imz{i} imz{i})")
    smt_lines.append(f"(assert (= (+ {' '.join(norm_terms)}) 1))")
    
    # Declare constant s representing √2.
    smt_lines.append("(declare-const s Real)")
    smt_lines.append("(assert (> s 0))")
    smt_lines.append("(assert (= (* s s) 2))")
    
    # Factor for H gate transformation: 1/(s^N)
    if N == 0:
        factor_str = "1"
    else:
        factor_str = f"(/ 1 (* {' '.join(['s' for _ in range(N)])}))"
    
    # Compute transformed amplitudes.
    # For each j from 0 to M-1, compute:
    #   z'_j_re = (1/(s^N)) * sum_{i=0}^{M-1} (-1)^(dot(i,j)) * z_i_re
    #   z'_j_im = (1/(s^N)) * sum_{i=0}^{M-1} (-1)^(dot(i,j)) * z_i_im
    transformed_re = {}
    transformed_im = {}
    for j in range(M):
        terms_re = []
        terms_im = []
        for i in range(M):
            sign = (-1)**bitwise_dot(i, j, N)
            terms_re.append(f"(* {sign} rez{i})")
            terms_im.append(f"(* {sign} imz{i})")
        sum_re = "(+ " + " ".join(terms_re) + ")"
        sum_im = "(+ " + " ".join(terms_im) + ")"
        transformed_re[j] = f"(* {factor_str} {sum_re})"
        transformed_im[j] = f"(* {factor_str} {sum_im})"
    
    # Assume B_expr is expressed in terms of the original variables z_i_re, z_i_im (i=0..M-1).
    # Define B_original as a function of these variables.
    B_str = str(B_expr)
    B_arg_decls = " ".join(f"(rez{i} Real) (imz{i} Real)" for i in range(M))
    smt_lines.append(f"(define-fun B_original ({B_arg_decls}) Real")
    smt_lines.append(f"  {B_str}")
    smt_lines.append(")")
    
    # Define B_transformed as B_original evaluated on the transformed amplitudes.
    # Build the argument list: for i = 0,..., M-1, pass (z'_i)_re and (z'_i)_im.
    B_transformed_args = []
    for i in range(M):
        B_transformed_args.append(transformed_re[i])
        B_transformed_args.append(transformed_im[i])
    B_transformed_args_str = " ".join(B_transformed_args)
    smt_lines.append(f"(define-fun B_transformed ({B_arg_decls}) Real")
    smt_lines.append(f"  (B_original {B_transformed_args_str})")
    smt_lines.append(")")
    
    # Build the universal quantifier over all original state variables.
    quant_vars = " ".join(f"(rez{i} Real) (imz{i} Real)" for i in range(M))
    original_vars_str = " ".join(f"rez{i} imz{i}" for i in range(M))
    
    # Assert dynamic constraint: for all state vectors, B_transformed - B_original <= 0.
    smt_lines.append(f"(assert ")
    smt_lines.append(f"  (<= (- (B_transformed {original_vars_str}) (B_original {original_vars_str})) 0)")
    smt_lines.append(")")
    
    smt_lines.append("(check-sat)")
    
    return "\n".join(smt_lines)

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

def make_constraints(conditions, constraints, var_z3_dict):
    # Loop over each constraint in Z0_constraints
    for constr in constraints:
        variables = constr['variables']
        min_sum = constr['min']
        max_sum = constr['max']
        # Sum over the modulus squared of the variables
        sum_modulus_squared = sum(
            var_z3_dict[sym.re(v)]**2 + var_z3_dict[sym.im(v)]**2 for v in variables
        )
        # Add the inequality constraints
        if min_sum is not None:
            conditions.append(sum_modulus_squared >= min_sum)
        if max_sum is not None:
            conditions.append(sum_modulus_squared <= max_sum)
        for k, v in constr['imConstr'].items():
            conditions.append(var_z3_dict[sym.im(k)] >= v[0])
            conditions.append(var_z3_dict[sym.im(k)]  <= v[1])
    return conditions

def check_barrier(Z:list[sym.Symbol],
                  barriers:list[sym.Poly],
                  unitaries,
                  Z0,
                  Zu,
                  d,
                  epsilon,
                  gamma,
                  K,
                  timeout=300,
                  log_level=logging.INFO):
    logger.setLevel(log_level)
    Z_RI = [sym.re(z) for z in Z] + [sym.im(z) for z in Z]
    var_z3_dict = dict(zip(Z_RI, [Real(str(var)) for var in Z_RI]))
    k = [v for v in var_z3_dict.values()]
    norm = Sum([var**2 for var in k]) == 1
    
    real_barrier = sym.poly(sym.re(sym.expand_complex(barriers[0].as_expr())))
    z3_barrier = _sympy_to_z3_rec(var_z3_dict, real_barrier.as_expr())
    initial_conditions = [norm, z3_barrier>0]
    initial_conditions = make_constraints(initial_conditions, Z0, var_z3_dict)
    logger.info("Checking initial constraint...")
    q = run_check(initial_conditions, timeout)
    if q:
        return q
    
    logger.info("checking unsafe constraints for barriers...")
    for barrier in barriers:
        real_barrier = sym.poly(sym.re(sym.expand_complex(barrier.as_expr())))
        z3_barrier = _sympy_to_z3_rec(var_z3_dict, real_barrier.as_expr())
        unsafe_conditions = [norm, z3_barrier<d-0.0001]
        unsafe_conditions = make_constraints(unsafe_conditions, Zu, var_z3_dict)
        q = run_check(unsafe_conditions, timeout)
        if q:
            return q
        
    if K > 1:
        logger.info("checking for dynamic constraint...")
        for unitary, barrier in zip(unitaries, barriers):
            fp = sym.poly(barrier.as_expr().subs(zip(Z, np.dot(unitary, Z)), simultaneous=True) - barrier)
            fp = sym.re((sym.expand_complex(fp.as_expr())))   
            z3_fx_barrier = _sympy_to_z3_rec(var_z3_dict, fp)
            dynamic_constraint = [norm, z3_fx_barrier > epsilon]
            q = run_check(dynamic_constraint, timeout)
            if q:
                return q
            
    if len(barriers) > 1:
        logger.info("checking for time evolution constraint...")
        fp = sym.poly(barriers[0] - barriers[1])
        fp = sym.re((sym.expand_complex(fp.as_expr())))
        fp1 = sym.poly(barriers[1] - barriers[0])
        fp1 = sym.re((sym.expand_complex(fp.as_expr())))
        z3_fx_barrier = _sympy_to_z3_rec(var_z3_dict, fp)
        z3_fx_barrier1 = _sympy_to_z3_rec(var_z3_dict, fp1)
        dynamic_constraint = [norm, z3_fx_barrier > gamma, z3_fx_barrier1 > gamma]
        q = run_check(dynamic_constraint, timeout)
        if q:
                return q
    
    logger.info("checking for k_induction constraint...")
    while(len(unitaries) < K): unitaries = unitaries + unitaries
    FZ = Z
    for i in range(K): FZ = np.dot(unitaries[i], FZ)
    fp = sym.poly(barriers[0].as_expr().subs(zip(Z, FZ), simultaneous=True) - barriers[0])
    fp = sym.re((sym.expand_complex(fp.as_expr())))
    z3_fx_barrier = _sympy_to_z3_rec(var_z3_dict, fp)
    dynamic_constraint = [norm, z3_fx_barrier > 0]
    q = run_check(dynamic_constraint, timeout)
    if q:
        return q
    
    return q

def run_check(conditions, timeout):
    s = z3.SolverFor('QF_NRA')
    set_option(precision=50)
    #set_option(timeout=10000)
    [s.add(cond) for cond in conditions]
    if s.check() == sat:
        m = s.model()
        logger.warning("Counterexample found: " +  str(m))
        return True   
    else:
        logger.info("No counterexamples found.")
        return False
    