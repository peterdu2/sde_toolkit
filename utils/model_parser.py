from cvxpy import diff
import yaml
import re


COEFF_IDX = 0
DEG_IDX = 1

def model_parser(filename):
    '''
    Function to parse yaml files containing an SDE model.

    Parameters:
        filename (str): YAML file containing model parameters
    Returns:
        state_dim (int): Dimension of state space
        states (list): List of state identifies (list of strings)
        drift_dynamics (dict): Dictionary containing coefficients and degrees of drift
        diffusion_dynamics (dict): Dictionary containing coefficients and degrees of diffusion

    Return value data format:
        drift_dynamics:
            -   Dictionary with entry for each state variable
            -   Each value entry contains a list of all monomials and
                their coefficients for the corresponding state variable
                E.g. {x: [[first monomial], [second monomial], ...]}
                where [first monomial] = [coefficient, [monomial degree]]

        diffusion dynamics:
            -   Dictionary with entry for each state variable
            -   Each value entry contains a list of all dynamics for each BM dimension
                where the dynamics are in the same format as the drift
                E.g. {x: [[diffusion for dBt_1], [diffusion for dBt_2], ...]},
                where [diffusion for dBt_x] = [[first monomial], [second monomial], ...],
                where [first monomial] = [coefficient, [monomial degree]]

        Example:
        To obtain the drift dynamics for the state variable 'var': drift_dynamics['var']
        To obtain the diffusion dynamics for the state variable 'var' for the 1st BM dimension: diffusion_dynamics['var'][0]
    '''
    stream = open(filename, 'r')
    model_data = yaml.load(stream, yaml.SafeLoader)

    assert 'solver' in model_data, 'Invalid model file: missing solver name'
    assert 'state_dim' in model_data, 'Invalid model file: missing state dimension'
    assert 'brownian_motion_dim' in model_data, 'Invalid model file: missing brownian motion dimension'
    assert 'state_vars' in model_data, 'Invalid model file: missing state variables'
    assert 'safe_set' in model_data, 'Invalid model file: missing safe set'
    assert 'safe_set_boundary' in model_data, 'Invalid model file: missing safe set boundary'
    assert 'initial_conditions' in model_data, 'Invalid model file: missing initial conditions'
    assert 'drift' in model_data, 'Invalid model file: missing drift dynamics'
    assert 'diffusion' in model_data, 'Invalid model file: missing diffusion dynamics'

    solver = model_data['solver']
    states = model_data['state_vars']
    state_dim = model_data['state_dim']
    bm_dim = model_data['brownian_motion_dim']
    safe_set_polys = model_data['safe_set']
    safe_set_boundary_poly = model_data['safe_set_boundary']
    ic = model_data['initial_conditions']
    drifts = model_data['drift']
    diffusions = model_data['diffusion']

    assert len(
        states) == model_data['state_dim'], 'Invalid model file: state dimension does not match given variable list'
    assert len(safe_set_polys) > 0, 'Invalid model file: empty safe set'
    assert len(safe_set_boundary_poly) > 0, 'Invalid model file: empty safe set boundary'
    for state in states:
        assert state in drifts, 'Missing drift dynamics for state: {}'.format(
            state)
        assert state in diffusions, 'Missing diffusion dynamics for state: {}'.format(
            state)

    # Parse safe set
    # Safe set consists of list of polynomials
    # Each polynomial is a list of monomial degrees and their coefficients: [[coeff, [degree]], ...]
    safe_set = []
    for ss_poly in safe_set_polys:
        cur_poly = []
        # Extract coefficients and monomial degree from polynomial
        for monomial in ss_poly.split('+'):
            num_data = re.findall(r'[-+]?(?:\d*\.\d+|\d+)', monomial)
            coeff = float(num_data[0])
            degree = [int(num_data[i]) for i in range(1, len(num_data))]
            cur_poly.append([coeff, degree])
        safe_set.append(cur_poly)
            

    # Parse safe set boundary
    # Safe set boundary consists of list of polynomials
    # Each polynomial is a list of monomial degrees and their coefficients: [[coeff, [degree]], ...]
    safe_set_boundary = []
    # Extract coefficients and monomial degree from polynomial
    for monomial in safe_set_boundary_poly[0].split('+'):
        num_data = re.findall(r'[-+]?(?:\d*\.\d+|\d+)', monomial)
        coeff = float(num_data[0])
        degree = [int(num_data[i]) for i in range(1, len(num_data))]
        safe_set_boundary.append([coeff, degree])


    # Drift dynamics stored in dictionary
    # Keys are state variables
    # Entries are lists of all monomial degrees and their coefficients: [[coeff, [degree]], ...]
    drift_dynamics = {}
    diffusion_dynamics = {}

    # Parse drift dynamics
    # Assign each monomial with appropriate coefficient
    for state in drifts:
        drift_dynamics[state] = []
        diffusion_dynamics[state] = [[] for i in range(bm_dim)]

        # Extract coefficients and monomial degree from drift
        for monomial in drifts[state].split('+'):
            # num_data[0]: coeff, num_data[1:]: monomial degree
            num_data = re.findall(r'[-+]?(?:\d*\.\d+|\d+)', monomial)
            coeff = float(num_data[0])
            degree = [int(num_data[i]) for i in range(1, len(num_data))]
            drift_dynamics[state].append([coeff, degree])

        # First split for each Brownian motion dimension
        for cur_bm_dim in range(bm_dim):
            cur_diffusion = diffusions[state].split(';')[cur_bm_dim]
            for monomial in cur_diffusion.split('+'):
                # num_data[0]: coeff, num_data[1:]: monomial degree
                num_data = re.findall(r'[-+]?(?:\d*\.\d+|\d+)', monomial)
                coeff = float(num_data[0])
                degree = [int(num_data[i]) for i in range(1, len(num_data))]
                diffusion_dynamics[state][cur_bm_dim].append([coeff, degree])

    # Generate squared diffusion dynamics in matrix form (sigma*simga.T)
    diffusion_matrix_squared = [
        [None for i in range(len(states))] for j in range(len(states))]
    for row in range(len(states)):
        for col in range(len(states)):
            cur_entry = []
            diffusion_row = diffusion_dynamics[states[row]]
            diffusion_col = diffusion_dynamics[states[col]]

            # Multiply each row by each col
            for bm_idx in range(len(states)):
                row_expression = diffusion_row[bm_idx]
                col_expression = diffusion_col[bm_idx]

                for row_entry in row_expression:
                    for col_entry in col_expression:
                        coefficient = row_entry[COEFF_IDX] * \
                            col_entry[COEFF_IDX]
                        degree = [sum(x) for x in zip(
                            row_entry[DEG_IDX], col_entry[DEG_IDX])]
                        if coefficient != 0.:
                            cur_entry.append([coefficient, degree])

            diffusion_matrix_squared[row][col] = cur_entry

    return solver, state_dim, states, bm_dim, safe_set, safe_set_boundary, ic, drift_dynamics, diffusion_dynamics, diffusion_matrix_squared
