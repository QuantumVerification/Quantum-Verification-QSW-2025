import numpy as np
from scipy.sparse import vstack, hstack
from sympy import Poly
from src.utils import *
from src.sampling import *
from src.check_barrier import *
from colorama import Fore, Style
import time
import subprocess
from multiprocessing import Process

def regMinus(Zu):
    ZminZu = []
    for i in range(len(Zu)):
        constr = {'variables': Zu[i]['variables']}
        if Zu[i]['min'] != 0:
            constr['min'] = 0
            constr['max'] = Zu[i]['min'] - 0.00000000001
        else:
            constr['min'] = Zu[i]['max'] + 0.00000000001
            constr['max'] = 1
        constr['imConstr'] = {}
        ZminZu.append(constr)
    return ZminZu



def run_cvc5(file_path):
    try:
        result = subprocess.run(
            ["cvc5", "--lang=smtlib2", file_path],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print("Risultato: ")
            print(result.stdout)
            return result.stdout
        else:
            print("Errore durante l'esecuzione di cvc5:")
            print(result.stderr)
            return None

    except subprocess.TimeoutExpired:
        print("L'esecuzione di cvc5 ha superato il tempo limite.")
        return None


# Function to check if a term contains both zi and conjugate(zi)
def is_real_term(term):
    from collections import defaultdict
    counts = defaultdict(int)
    for elem in term:
        # Usa le stringhe per gestire i simboli e i loro coniugati
        elem_str = str(elem)
        conjugate_str = str(conjugate(elem))

        if conjugate_str in counts:
            counts[conjugate_str] -= 1
        else:
            counts[elem_str] += 1
    # Un termine è reale se tutti i conteggi sono zero
    return all(count == 0 for count in counts.values())


def sort_terms(terms):
    sorted_terms = sorted(
        terms,
        key=lambda term: (
            len(term) != 0,  # La tupla vuota al primo posto
            -len(term),  # Termini di grado più basso prima
            (0 if len(term) % 2 == 0 and is_real_term(term) else 1),  # Termini reali di grado pari prima
            [str(e) for e in term]  # Ordine alfabetico per default
        )
    )
    return sorted_terms

def generate_values(i_samples,u_samples, d_samples, dynamic_samples, dynamic_k_samples, terms):

    #print("Sampling...")
    '''
    i_samples = sample_states(Z0, Z, n_samples)
    u_samples = sample_states(Zu, Z, n_samples)
    d_samples = sample_states([], Z, n_samples)


    print("fase 2...")
    state_vectors = create_state_vectors(d_samples, Z)
    dynamic_samples = generate_fx_samples(state_vectors, unitary, Z)
    dynamic_k_samples = generate_fx_samples_k_times(state_vectors, unitary, Z, k)

    '''

    print("fase 3...")
    i_sampled_terms = generate_sampled_terms(terms, i_samples)
    u_sampled_terms = generate_sampled_terms(terms, u_samples)
    d_sampled_terms = generate_sampled_terms(terms, d_samples)
    dynamic_sampled_terms = generate_sampled_terms(terms, dynamic_samples)
    dynamic_k_sampled_terms = generate_sampled_terms(terms, dynamic_k_samples)


    print("fase 4...")
    i_values = separate_real_imag(i_sampled_terms)
    u_values = separate_real_imag(u_sampled_terms)
    d_values = separate_real_imag(d_sampled_terms)
    dynamic_values = separate_real_imag(dynamic_sampled_terms)
    dynamic_k_values = separate_real_imag(dynamic_k_sampled_terms)

    return i_values, u_values, dynamic_values, d_values, dynamic_k_values

def sample(Z, Z0, Zu, unitary, n_samples, k):
    i_samples = sample_states(Z0, Z, n_samples)
    u_samples = sample_states(Zu, Z, n_samples)
    d_samples = sample_states([], Z, n_samples)


    print("fase 2...")
    state_vectors = create_state_vectors(d_samples, Z)
    dynamic_samples = generate_fx_samples(state_vectors, unitary, Z)
    dynamic_k_samples = generate_fx_samples_k_times(state_vectors, unitary, Z, k)

    return i_samples, u_samples, d_samples, dynamic_samples, dynamic_k_samples


def verifyQC(unitary, n_samples, Z, Z0, Zu, poly_degree, opt_meth, k):
    gen_time = []
    ver_time = []
    for i in range(10):
        start_gen = time.time()

        var = generate_variables(Z)
        terms2 = generate_terms(var, poly_degree)
        terms2 = sort_terms(terms2)
        #terms = terms2[:9]



        print("TOTAL TERMS: " ,terms2)

        #terms = [terms2[0],terms2[1], terms2[2]]
        #terms = [terms2[i] for i in range(7)]
        num_terms = 0
        terms  = []
        for term in terms2:
            print("Sampling...")
            i_samples, u_samples, d_samples, dynamic_samples, dynamic_k_samples = sample(Z, Z0, Zu, unitary, n_samples, k)

            start = time.time()

            num_terms += 1

            terms.append(term)

            #terms = terms2[:9]
            print(Style.RESET_ALL + "TERMS: " , terms)
            print("Number of terms: ", num_terms)


            l = len(terms)
            num_coefficients = 2 * l + 2
            c = np.zeros(num_coefficients)
            c[-1] = -1

            print(f'Genero Values')
            i_values, u_values, dynamic_values, d_values, dynamic_k_values = generate_values(i_samples,u_samples, d_samples, dynamic_samples, dynamic_k_samples, terms)

            print(f'Genero Conditions')
            Aub, bub, Aeq, beq = generate_all_constraints(i_values, u_values, dynamic_values, d_values, dynamic_k_values, k, constr=True)
            #print_linprog_problem(Aub, bub, Aeq, beq, c)


            coeff_bounds = np.inf
            bounds = [(-coeff_bounds, coeff_bounds)] * num_coefficients
            y_upper_bound = 5
            bounds[-1] = (0.000001, y_upper_bound)
            i = n_samples * 3

            print(Style.RESET_ALL + f'Numero di campioni: {i} ')
            # start_gen = time.time()
            barrier_certificate, a, y, epsilon = (solve_lp
                                                  (c, Aub, bub, None, None, bounds, l, terms, var, opt_meth))
            end_gen = time.time()
            print(Fore.GREEN + f'TEMPO DI GENERAZIONE: {end_gen - start_gen} ')



            if a is None:
                print(
                    "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                continue
            print(Fore.BLUE + "Certificato candidato: ", barrier_certificate)
            print()

            if barrier_certificate is None:
                print("NO")
                continue
            else:
                gen_time.append(round(end_gen - start_gen, 4))
                start_ver = time.time()
                print(f'Epsilon: {epsilon}')
                init_add_sample, unsafe_add_sample, d_add_sample = check_barrier(Z, barrier_certificate, unitary, var, Z0,
                                                                                 Zu, k, epsilon, y)
            end_ver = time.time()
            print(Fore.GREEN + f'Tempo di Verifica: {end_ver - start_ver}')
            print(init_add_sample, unsafe_add_sample, d_add_sample)
            ver_time.append(round(end_ver - start_ver, 4))

            break
            '''
            
            while True:
                print(Style.RESET_ALL + f'Numero di campioni: {i} ')
                #start_gen = time.time()
                barrier_certificate, a, y, epsilon = (solve_lp
                                             (c, Aub, bub, None, None, bounds, l, terms, var, opt_meth))
                end_gen = time.time()
                print(Style.RESET_ALL + f'TEMPO DI GENERAZIONE: {end_gen - start_gen} ')
    
                if a is None:
                    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                    break
                print(Fore.BLUE + "Certificato candidato: ", barrier_certificate)
                print()
    
    
                if barrier_certificate is None:
                    print("NO")
                    break
                else:
                    start_ver = time.time()
                    print(f'Epsilon: {epsilon}')
                    init_add_sample, unsafe_add_sample, d_add_sample = check_barrier(Z, barrier_certificate, unitary, var, Z0, Zu, k, epsilon, y)
    
                    if init_add_sample == [{}] and unsafe_add_sample == [{}] and d_add_sample == [{}]:
                        end_ver = time.time()
    
                        executionTime = time.time() - start
                        print(Fore.GREEN + "Polinomio confermato: ", barrier_certificate)
                        print()
                        print(f'Execution time: {executionTime}')
                        print(f'TEMPO DI VERIFICA: {end_ver - start_ver}')
                        print()
                        print(Fore.LIGHTMAGENTA_EX + "Verifica formale CVC5: ")
                        print()
    
                        
                        p0 = Process(target=run_cvc5, args=("./smtfiles/output0.smt2",))
                        p1 = Process(target=run_cvc5, args=("./smtfiles/output1.smt2",))
                        p2 = Process(target=run_cvc5, args=("./smtfiles/output2.smt2",))
    
                        p0.start()
                        p1.start()
                        p2.start()
    
                        p0.join()
                        p1.join()
                        p2.join()
    
    
                        exit()
    
                    #ival, uval, dynval, dval = generate_values(Z, Z0, Zu, unitary, terms, 100)
                    i_values, u_values, dynamic_values, d_values, dynamic_k_values = None, None, None, None, None
    
                    if init_add_sample != [{}]:
                        print("INIT ADD SAMPLE ", init_add_sample)
                        i += 1
                        i_sampled_terms = generate_sampled_terms(terms, init_add_sample)
                        i_values = separate_real_imag(i_sampled_terms)
                        #i_values = np.concatenate((i_values, ival))
    
                    if unsafe_add_sample != [{}]:
                        print("INIT ADD SAMPLE ", unsafe_add_sample)
                        i += 1
                        u_sampled_terms = generate_sampled_terms(terms, unsafe_add_sample)
                        u_values = separate_real_imag(u_sampled_terms)
                        #u_values = np.concatenate((u_values, uval))
                    if  d_add_sample != [{}]:
                        i += 1
                        state_vector = create_state_vectors(d_add_sample, Z)
                        dynamic_sample = generate_fx_samples(state_vector, unitary, Z)
                        dynamic_k_sample = generate_fx_samples_k_times(state_vector, unitary, Z,k)
    
                        d_sampled_terms = generate_sampled_terms(terms, d_add_sample)
                        dynamic_sampled_terms = generate_sampled_terms(terms, dynamic_sample)
                        dynamic_k_sampled_terms = generate_sampled_terms(terms, dynamic_k_sample)
                        d_values = separate_real_imag(d_sampled_terms)
                        dynamic_values = separate_real_imag(dynamic_sampled_terms)
                        dynamic_k_values = separate_real_imag(dynamic_k_sampled_terms)
                        #dynamic_values = np.concatenate((dynamic_values, dynval))
                        #d_values = np.concatenate((d_values, dval))
    
    
                    Aub2, bub2, Aeq2, beq2 = generate_all_constraints(i_values, u_values, dynamic_values, d_values, dynamic_k_values, k)
                    Aub = np.concatenate((Aub, Aub2), axis=0)
                    bub = np.hstack([bub, bub2])
                    #print_linprog_problem(Aub, bub, Aeq, beq, c)
            '''
    mean_gen = np.mean(gen_time)
    mean_ver = np.mean(ver_time)

    std_gen = np.std(gen_time)
    std_ver = np.std(ver_time)

    print(f'Mean Generation: {mean_gen}')
    print(f'Mean Verification: {mean_ver}')
    print(f'Standard Deviation: {std_gen}')
    print(f'Standard Deviation: {std_ver}')








