import cvxpy as cp
import numpy as np
from functools import cmp_to_key

from utils.multi_index import generate_unsorted_idx_list, index_comparison


COEFF_IDX = 0
DEG_IDX = 1

def is_lm_valid(degree, lm_dim, moment_list):
    '''
    Check if the given size of localizing matrix (lm) can be
    supported by a given moment list.

    Parameters:
        degree (list): multi index of highest monomial added to lm entry
        lm_dim (int): size of localizing matrix
        moment_list (list(list)): available moments
    Returns:
        (bool): if moment_list contains all terms required for desired lm
    '''
    for i in range(lm_dim):
        for j in range(lm_dim):
            mm_moment = [sum(x) for x in zip(
                moment_list[i], moment_list[j])]
            lm_entry = [sum(x) for x in zip(mm_moment, degree)]
            if not lm_entry in moment_list:
                return False
    return True

def create_moment_vars(n_moments):
    '''
    Create CVXPY variables for moments

    Parameters:
        n_moments (int): number of moments
    Returns:
        (list): list of CVXPY variables for all moments
    '''
    return [cp.Variable() for i in range(n_moments)]

def create_moment_matrix_vars(mm_degree, state_dim):
    '''
    Create CVXPY variable for moment matrix

    Parameters:
        mm_degree (int): order of moment matrix
        state_dim (int): dimension of state
    Returns:
        (cvxpy.Variable): CVXPY PSD matrix variable
    '''
    idx_list = generate_unsorted_idx_list(M=mm_degree, d=state_dim)
    return cp.Variable((len(idx_list), len(idx_list)), PSD=True)

def create_localizing_matrix_vars(polynomials, max_moment_degree, max_poly_degree, state_dim, moment_list):
    '''
    Create CVXPY variables for localizing matrices

    Parameter:
        polynomials (list): list of polynomials defining semi-algebraic set
        max_moment_degree (int): maxmimum moment degree of optimization (K)
        max_poly_degree (int): maxmimum degree of polynomial
        state_dim (int): dimension of state space
        moment_list (list): available moments used for optimization
    Returns:
        (list(cvxpy.Variable)): list of CVXPY PSD matrix variables
    '''
    # Generate a sorted list of multi-indicies
    idx_list = generate_unsorted_idx_list(M=max_poly_degree, d=state_dim)
    idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

    lms = []    # List to hold all localizing matrix opt variables

    for polynomial in polynomials:
        # Create full length coefficient list from polynomial
        cur_poly_coeffs = np.zeros(len(idx_list)).tolist()
        for monomial in polynomial:
            cur_poly_coeffs[idx_list.index(monomial[DEG_IDX])] = monomial[COEFF_IDX]
        
        # Find the largest index (monomial) in the polynomial that has non-zero coefficient
        for k in range(len(idx_list)-1, -1, -1):
            if cur_poly_coeffs[k] != 0:
                break
        
        # Find the largest localizing matrix that can be supported by the defined moment sequence
        for i in range(max_moment_degree):
            if not is_lm_valid(idx_list[k], len(generate_unsorted_idx_list(M=i, d=state_dim)), moment_list):
                break
        
        lm_dim = len(generate_unsorted_idx_list(M=i-1, d=state_dim))
        lms.append(cp.Variable((lm_dim, lm_dim), PSD=True))

    return lms
