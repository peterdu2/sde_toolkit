import cvxpy as cp
import numpy as np
from functools import cmp_to_key

import utils.cvxpy_utils as cvxpy_utils
from utils.model import SDEModel
from utils.multi_index import generate_unsorted_idx_list, index_comparison, mono_derivative

COEFF_IDX = 0
DEG_IDX = 1

class OptimizationProgram:
    '''
    Class for optimization program computing SDE exit time moments.

    Attributes
    ----------
        model (SDEModel): SDE dynamics model
        max_degree (int): maxmimum order of moments used in optimization (K)
    '''
    def __init__(self, model, max_degree):
        self.model = model
        self.max_degree = max_degree

        # Moments are identified by their degree (given by list)
        # e.g. M_123 = [1,2,3]
        moment_list = generate_unsorted_idx_list(
            M=max_degree, d=model.state_dim)
        self.moment_list = sorted(
            moment_list, key=cmp_to_key(index_comparison))

        # Optimization variables for moments of occupation and exit measure
        self.mj = None
        self.bj = None

        # Optimization variables for moment matrices
        self.mm_mj = None
        self.mm_bj = None
        self.mm_dict = {}   # Data structure to keep track of what each moment index
        # is present in each entry of the moment matrices

        # Optimization variables for localizing matrix
        self.lm_mj = None


    def create_opt_vars(self):
        '''
        Create all optimization variables.
        '''
        # Currently only supports CVXPY
        if self.model.solver == 'cvxpy':
            # Create variables for the moments of occupation (mj) and exit (bj) measures
            self.mj = cvxpy_utils.create_moment_vars(len(self.moment_list))
            self.bj = cvxpy_utils.create_moment_vars(len(self.moment_list))

            # Create variables for the moment matrices of the occupation (mm_mj) and exit (mm_bj) measures
            max_mm_degree = self.num_moment_matrices()-1
            self.mm_mj = cvxpy_utils.create_moment_matrix_vars(
                max_mm_degree, self.model.state_dim)
            self.mm_bj = cvxpy_utils.create_moment_matrix_vars(
                max_mm_degree, self.model.state_dim)

            # Create variable for the localizing matrix of the occupation measure
            self.lm_mj = cvxpy_utils.create_localizing_matrix_vars(
                self.model.safe_set, self.max_degree, self.max_degree, self.model.state_dim, self.moment_list)


    def num_moment_matrices(self):
        '''
        Compute maximum number of moment matrices that can be created
        given the maximum moment degree (K) of the optimization. 
        '''
        for mm_idx in range(self.max_degree):
            idx_list = generate_unsorted_idx_list(
                M=mm_idx, d=self.model.state_dim)
            idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
            if len(idx_list) > len(self.moment_list):
                return mm_idx
            for i in range(len(idx_list)):
                for j in range(len(idx_list)):
                    moment_idx = [sum(x) for x in zip(
                        self.moment_list[i], self.moment_list[j])]
                    if moment_idx not in self.moment_list:
                        return mm_idx
        return mm_idx+1


    def create_mm_constraints(self):
        '''
        Create constraints that connects entries of the moment matrices with
        the corresponding entries in the list of moment optimization variables.

        Returns:
            constraints (list): list of CVXPY constraints
        '''
        # Generate list of moment matrix constraints
        # Assign appropriate moments to each moment matrix entry
        # All constraints have format: mm[i][j] = moment_x
        constraints = []

        assert self.mm_mj.shape[0] == self.mm_bj.shape[0], 'Moment matrix dimensions for occupation and exit do not match'
        assert self.mm_mj.shape[1] == self.mm_bj.shape[1], 'Moment matrix dimensions for occupation and exit do not match'

        # Loop through every entry of each moment matrix
        for i in range(self.mm_mj.shape[0]):
            for j in range(self.mm_mj.shape[1]):
                # Compute the moment index at this entry of the moment matrices
                moment_idx = [sum(x) for x in zip(
                    self.moment_list[i], self.moment_list[j])]
                # Generate equality constraint
                constraints.append(self.mm_mj[i][j]
                                   == self.mj[self.moment_list.index(moment_idx)])
                constraints.append(self.mm_bj[i][j]
                                   == self.bj[self.moment_list.index(moment_idx)])

                # Create dictionary entry for keep track of which moments are present at
                # each entry in the moment matrices (Used for determining localizing matrices)
                self.mm_dict[str(i)+'_'+str(j)] = moment_idx

        return constraints


    def create_lm_constraints(self):
        '''
        Create constraints that connects entries of the localizing matrices with
        the corresponding entries in the list of moment optimization variables.

        Returns:
            constraints (list): list of CVXPY constraints
        '''
        # Assign appropriate moments to each localizing matrix entry
        # All constraints have format: lm[i][j] = f(moment_x)
        constraints = []

        # Loop through every localizing matrix
        for cur_lm in range(len(self.lm_mj)):
            # Link each entry of the localizing matrix to the appropriate combination of moments
            for i in range(self.lm_mj[cur_lm].shape[0]):
                for j in range(self.lm_mj[cur_lm].shape[1]):
                    # Get the moment index of the corresponding entry in the moment matrix
                    mm_idx = self.mm_dict[str(i)+'_'+str(j)]

                    # Create full length coefficient list from polynomial
                    cur_poly_coeffs = np.zeros(len(self.moment_list)).tolist()
                    for monomial in self.model.safe_set[cur_lm]:
                        cur_poly_coeffs[self.moment_list.index(
                            monomial[DEG_IDX])] = monomial[COEFF_IDX]

                    constraint_ij = 0.
                    for alpha in range(len(cur_poly_coeffs)):
                        # Get moment index by summing alpha index with moment matrix index
                        lm_idx = [sum(x) for x in zip(
                            self.moment_list[alpha], mm_idx)]
                        if cur_poly_coeffs[alpha] != 0:
                            constraint_ij += cur_poly_coeffs[alpha] * \
                                self.mj[self.moment_list.index(lm_idx)]

                    constraints.append(
                        constraint_ij == self.lm_mj[cur_lm][i][j])

        return constraints

    def create_lm_eq_constraints(self):
        '''
        Create constraints for reformulated localizing matrix equality constriants 
        using entries in the list of moment optimization variables.

        Returns:
            constraints (list): list of CVXPY constraints
        '''
        # Generate list of exit measure localizing matrix equality constraints
        # Compute the entry of the corresponding localizing matrix and set to 0
        # All constraints have format: lm_exit_measure[i][j] = f(bj's) = 0
        constraints = []

        # Create full length coefficient list from polynomial
        poly_coeffs = np.zeros(len(self.moment_list)).tolist()
        for monomial in self.model.safe_set_boundary:
            poly_coeffs[self.moment_list.index(
                monomial[DEG_IDX])] = monomial[COEFF_IDX]

        # Find the largest index in the polynomial coeff that isn't 0
        for k in range(len(self.moment_list)-1, -1, -1):
            if poly_coeffs[k] != 0:
                break

        # Find the size of the largest localizing matrix that can be supported by the defined moment sequence
        for i in range(self.max_degree):
            if not cvxpy_utils.is_lm_valid(self.moment_list[k],
                                           len(generate_unsorted_idx_list(
                                               M=i, d=self.model.state_dim)),
                                           self.moment_list):
                break
        lm_dim = len(generate_unsorted_idx_list(M=i-1, d=self.model.state_dim))

        # Construct localising matrix entries and set to zero (equality constriants)
        for i in range(lm_dim):
            for j in range(lm_dim):
                # Get the moment index of the corresponding entry in the moment matrix
                mm_idx = self.mm_dict[str(i)+'_'+str(j)]

                constraint_ij = 0.
                # Sum over all coefficients present in safe set boundary polynomial
                for alpha in range(len(poly_coeffs)):
                    if poly_coeffs[alpha] != 0:
                        # Get moment index by summing alpha index with moment matrix index
                        lm_idx = [sum(x) for x in zip(
                            self.moment_list[alpha], mm_idx)]
                        constraint_ij += poly_coeffs[alpha] * \
                            self.bj[self.moment_list.index(lm_idx)]

                constraints.append(
                    constraint_ij == 0)

        return constraints

    def create_mt_constraints(self):
        '''
        Create martingale (evolution) equality constraints for monomial test functions
        up to degree K (max moment degree). 

        Returns:
            constraints (list): list of CVXPY constraints
        '''
        # Generate list of martingale constraints
        constraints = []

        # Monomial test functions up to K = self.max_degree is considered
        # split into three cases: deg(f) = 0, deg(f) = 1, deg(f) > 1

        # deg(f) = 0
        # Test function 'f' is a constant (infinitesimal generator = 0)
        constraints.append(-self.bj[0]+1 == 0)

        # deg(f) = 1
        # Test function 'f' is a monomial of degree 1. Infinitesimal generator only
        # contains drift terms
        for f in self.moment_list[1:]:
            if not np.sum(f) == 1:
                break
            else:
                # Compute infinitesimal generator
                # Af(x) = \sum_i{h_i * df/ds_i} for each state s_i
                generator = []
                for state in self.model.states:
                    df_dstate = mono_derivative(
                        f, [self.model.states.index(state)])
                    if df_dstate[COEFF_IDX] != 0.:
                        for drift_term in self.model.drift_dynamics[state]:
                            generator_term = [
                                drift_term[COEFF_IDX] * df_dstate[COEFF_IDX]]
                            generator_term.append([sum(x) for x in zip(
                                drift_term[DEG_IDX], df_dstate[DEG_IDX])])
                            generator.append(generator_term)

                # Check if all terms in generator can be represented with given moment sequence
                invalid_f = False
                for monomial in generator:
                    if not monomial[DEG_IDX] in self.moment_list:
                        invalid_f = True
                        break

                # Assemble constraint expression
                if not invalid_f:
                    constraint_f = 0.
                    # Add generator terms (m_j's)
                    for monomial in generator:
                        if monomial[COEFF_IDX] != 0:
                            # Match monomial term with corresponding moment (optimization) variable
                            constraint_f += monomial[COEFF_IDX] * \
                                self.mj[self.moment_list.index(monomial[DEG_IDX])]
                    # Add initial condition f(s0)
                    s0 = 1.0
                    for i in range(len(f)):
                        if f[i] != 0:
                            s0 *= self.model.initial_conditions[i]**f[i]
                    constraint_f += s0
                    # Add exit measure (b_j's)
                    constraint_f += -self.bj[self.moment_list.index(f)]

                    # Add completed constraint to list
                    constraints.append(constraint_f == 0)

        # deg(f) > 1
        # Test function 'f' is a monomial of degree > 1. Infinitesimal generator
        # contains both drift and diffusion terms
        for f in self.moment_list:
            if not np.sum(f) > 1:
                pass
            else:
                # Compute infinitesimal generator
                # Af(x) = sum_i{h_i * df/ds_i} + 0.5 * sum_ij{oo.T_ij * d^2f/ds_i ds_j}
                # for each state s_i
                generator = []

                # Compute drift portion of Af(x)
                for state in self.model.states:
                    df_dstate = mono_derivative(
                        f, [self.model.states.index(state)])
                    if df_dstate[COEFF_IDX] != 0.:
                        for drift_term in self.model.drift_dynamics[state]:
                            generator_term = [
                                drift_term[COEFF_IDX] * df_dstate[COEFF_IDX]]
                            generator_term.append([sum(x) for x in zip(
                                drift_term[DEG_IDX], df_dstate[DEG_IDX])])
                            generator.append(generator_term)

                # Compute diffusion portion of Af(x)
                for i in range(self.model.state_dim):
                    for j in range(self.model.state_dim):
                        d2f_dstate2 = mono_derivative(f, [i, j])
                        if d2f_dstate2[COEFF_IDX] != 0.:
                            for term in self.model.diffusion_matrix_squared[i][j]:
                                generator_term = [
                                    0.5 * term[COEFF_IDX] * d2f_dstate2[COEFF_IDX]]
                                generator_term.append([sum(x) for x in zip(
                                    term[DEG_IDX], d2f_dstate2[DEG_IDX])])
                                generator.append(generator_term)

                # Check if all terms in generator can be represented with given moment sequence
                invalid_f = False
                for monomial in generator:
                    if not monomial[DEG_IDX] in self.moment_list:
                        invalid_f = True
                        break

                # Assemble constraint expression
                if not invalid_f:
                    constraint_f = 0.
                    # Add generator terms (m_j's)
                    for monomial in generator:
                        if monomial[COEFF_IDX] != 0:
                            # Match monomial term with corresponding moment (optimization) variable
                            constraint_f += monomial[COEFF_IDX] * \
                                self.mj[self.moment_list.index(monomial[DEG_IDX])]
                    # Add initial condition f(s0)
                    s0 = 1.0
                    for i in range(len(f)):
                        if f[i] != 0:
                            s0 *= self.model.initial_conditions[i]**f[i]
                    constraint_f += s0
                    # Add exit measure (b_j's)
                    constraint_f += -self.bj[self.moment_list.index(f)]

                    # Add completed constraint to list
                    constraints.append(constraint_f == 0)

        return constraints
