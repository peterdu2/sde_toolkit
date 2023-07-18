import cvxpy as cp
import sys

from utils.model import SDEModel
from utils.config_parser import config_parser
from utils.optimization import OptimizationProgram

TIME_IDX = 1

'''
Example runner script to generate moment optimization program and record results in a .txt file.
This runner uses splitting conic solver (SCS) from CVXPY optimization library.
'''


if __name__ == '__main__':
    # Parse problem configuration file
    config_file = 'configs/example_config.yaml'
    config_data = config_parser(config_file)
    
    assert config_data['solver'] == 'scs', 'Unsupported solver, example runner only supports SCS'

    # Create SDE model object
    model = SDEModel(config_data['model_path'] + config_data['model_filename'])

    # Prase optional parameters for SCS solver
    solver_eps = config_data['optional_solver_params']['solver_eps']
    solver_ac = config_data['optional_solver_params']['solver_ac']
    solver_scale = config_data['optional_solver_params']['solver_scale']
    solver_max_iters = config_data['optional_solver_params']['solver_max_iters']

    # Run optimization for each exit time moment to be calculated
    exit_time_moments = config_data['exit_time_moments']
    results = []
    for moment_order in exit_time_moments:
        # Create optimization program and constraint list
        opt = OptimizationProgram(model, max_degree=config_data['K'])
        opt.create_opt_vars()
        constraints = []
        constraints += opt.create_mm_constraints()
        constraints += opt.create_lm_constraints()
        constraints += opt.create_lm_eq_constraints()
        constraints += opt.create_mt_constraints()

        # Calculate index of the moment being computed
        moment_idx = [0 for m in range(model.state_dim)]
        moment_idx[TIME_IDX] = moment_order - 1
        
        # Generate objective and run optimization
        obj = cp.Maximize(moment_order*opt.mj[opt.moment_list.index(moment_idx)]) if config_data['mode'] == 'maximize' \
                          else cp.Minimize(moment_order*opt.mj[opt.moment_list.index(moment_idx)])
        prob = cp.Problem(obj, constraints)
        prob.solve(max_iters=solver_max_iters, verbose=True, solver=cp.SCS)

        results.append(prob.value)
        print(prob.value)
        print(prob.status)

        # Record output into text file
        original_stdout = sys.stdout
        filename = 'example_output.txt'
        with open(filename, 'a') as f:
            sys.stdout = f
            print('##################################################################')
            print('mode:', config_data['mode'])
            print('exit time moment:', moment_order)
            print('value:', prob.value)
            print('status:', prob.status)
            print('solver:', config_data['solver'])
            print('solver max iterations:', solver_max_iters)
            print('solver eps:', solver_eps)
            print('solver scale:', solver_scale)
            print('States:', model.states)
            print('Initial state:', model.initial_conditions)
            print('Safe set:', model.safe_set)
            print('Safe boundary:', model.safe_set_boundary)
            print('##################################################################\n')
            sys.stdout = original_stdout

    print(results)