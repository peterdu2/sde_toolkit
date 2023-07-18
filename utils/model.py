from utils.model_parser import model_parser


class SDEModel:
    def __init__(self, filename):
        solver, dim, states, bm_dim, safe_set, safe_set_boundary, ic, drift, diffusion, diffusion_matrix_squared = model_parser(
            filename)
        self.solver = solver
        self.state_dim = dim
        self.states = states
        self.bm_dim = bm_dim
        self.safe_set = safe_set
        self.safe_set_boundary = safe_set_boundary
        self.initial_conditions = ic
        self.drift_dynamics = drift
        self.diffusion_dynamics = diffusion
        self.diffusion_matrix_squared = diffusion_matrix_squared
