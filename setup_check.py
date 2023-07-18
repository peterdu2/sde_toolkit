import cvxpy as cp
import sys

from utils.model import SDEModel
from utils.config_parser import config_parser
from utils.optimization import OptimizationProgram

if __name__ == '__main__':
    config_file = 'configs/example_config.yaml'
    config_data = config_parser(config_file)
    assert config_data['solver'] == 'scs', 'Unsupported solver, example runner only supports SCS'

    model = SDEModel(config_data['model_path'] + config_data['model_filename'])

    moment_order = 1
    opt = OptimizationProgram(model, max_degree=config_data['K'])
    opt.create_opt_vars()
    constraints = []
    constraints += opt.create_mm_constraints()
    constraints += opt.create_lm_constraints()
    constraints += opt.create_lm_eq_constraints()
    constraints += opt.create_mt_constraints()
    moment_idx = [0 for m in range(model.state_dim)]
    obj = cp.Maximize(moment_order*opt.mj[opt.moment_list.index(moment_idx)]) if config_data['mode'] == 'maximize' \
                          else cp.Minimize(moment_order*opt.mj[opt.moment_list.index(moment_idx)])
    prob = cp.Problem(obj, constraints)

    print('Required dependencies ok')