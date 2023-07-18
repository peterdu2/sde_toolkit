import numpy as np
from functools import cmp_to_key
import copy

def sums(length, total_sum):
    '''
    Generate permuations
    '''
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value):
                yield (value,) + permutation


def generate_unsorted_idx_list(M, d):
    '''
    Create an unsorted list of all multi-indicies up to give degree.

    Parameters:
        M (int): maximum degree of multi-index
        d (int): state dimension
    
    Returns:
        idx_list (list): unsorted list of multi-indicies
    '''
    idx_list = [[0 for x in range(d)]]
    for cur_degree in range(0, M+1):

        # Generate all permutations of dimension d
        L = list(sums(d, cur_degree))

        for idx in L:
            idx_entry = [x for x in idx]
            if idx_entry not in idx_list:
                idx_list.append(idx_entry)

    return idx_list


'''
Input:
	Two multi indicies represetned as lists
Output:
	True if item1 < item2
'''
def index_comparison(item1, item2):
    '''
    Compare two multi-indicies according to graded lexicographic order

    Parameters:
        item1 (list): first multi-index
        item2 (list): second multi-index
    Returns:
        (int): -1 if item1 precedes item2, else 1
    '''
    # If deg(item1) < deg(item2) then item1 < item2
    # If deg(item1) == deg(item2):
    #	The larger element is the one whose exponent vector (multi-index) is lexically smaller.
    #	From left to right, first element that differs, the vector with higher exponent
    #	is the one smaller lexically
    # 	Ref: https://people.sc.fsu.edu/~jburkardt/m_src/polynomial/polynomial.html

    if sum(item1) != sum(item2):
        return -1 if sum(item1) < sum(item2) else 1

    for i in range(0, len(item1)):
        if item1[i] != item2[i]:
            return -1 if item1[i] > item2[i] else 1


def mono_derivative(monomial, dx_list):
    '''
    Take the derivative of a monomial.

    Parameters:
        monomial (list): monomial to differentiate
        dx_list (list): number of times to differentiante each term in monomial
    Returns:
        (list): coefficients and degrees of monomial after differentiation

    Example:
        monomial: [2, 4]
        dx_list: [0, 1]
        returns: [4, [2, 3]]
    '''
    coeff = 1
    degrees = copy.copy(monomial)

    for deriv_var in dx_list:

        if degrees[deriv_var] == 0:  # Check if taking derivative will result in zero
            return [0, []]
        else:
            coeff *= degrees[deriv_var]
            degrees[deriv_var] -= 1

    return [coeff, degrees]
