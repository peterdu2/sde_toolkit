# SDE Toolkit

This repository contains a set of Python tools for specifying and running semidefinite programs to compute upper and lower bounds on exit-time moments of stochastic dynamics. For details on the moment-based method for safety analysis, please refer to the paper [here](https://arxiv.org/abs/2104.04892).


## Setup
1. Install Python 3.8 (The code may work with other versions of Python, but 3.8 is recommended).

2. Install the required python dependencies. If using conda, install each package in `requirements.txt` into your conda environment manually. If using pip, install the dependencies with the following command:  
```
pip install -r requirements.txt
```

3. Run the provided `setup_check.py` file to verify that all dependencies are correct and accessible by the toolkit.



## Getting started
This repository is organized in three sections: 
- `configs/` contains YAML files for configurating parameters used by the optimization program. 
- `models/` contains YAML files which specify stochastic differential equation (SDE) models for each system under test. 
- `utils/` contains the source code for parsing model files, creating a semidefinite program, and running the optimization. 

### Specifying Stochastic Differential Equation Models
Models are specified using YAML files. Each model should contain the following:
- `solver` - String specifying the solver for the optimization program.

- `state_dim` - Integer specifying the dimensionality of the state space.

- `brownian_motion_dim` - Integer specifying the dimensionality of the Brownian motion driving the stochastic dynamics.

- `state_vars` - List of strings that provide a unique identifier to each state dimension. Number of list members must match `state_dim`.

- `initial_conditions` - List of floats specifying the initial condition of each state dimension. Number of list members must match `state_dim`.

- `safe set` - List of strings specifying the semialgebraic safe set. Each member is a polynomial consisting of monomial terms where each monomial contains a leading coefficient (float) and entries for all state dimensions (see `models/brownian_motion.yaml` for example).

- `safe_set_boundary` - String specifying the boundary of the safe set. The polynomial consists of monomial terms where each monomial contains a leading coefficient (float) and entries for all state dimensions (see `models/brownian_motion.yaml` for example).

- `drift` - List of strings specifying polynomials for the deterministic drift dynamics of the SDE (see `models/brownian_motion.yaml` for example).

- `diffusion` - List of strings specifying the polynomials for the diffusion dynamics of the SDE. Polynomials for each dimension of the Brownian motion are separated by a semicolon (see `models/brownian_motion.yaml` for example). 


### Creating a Runner
A runner file is required to parse the SDE models and create/run an optimization program. An example of a runner file is provided in `example_runner.py`. Before creating a runner you must have 1) a configuration YAML file, 2) a model YAML file. 

You may create your own runner by using the following guideline:

1. Instantiate an `SDEModel` object by using the `utils.model.SDEModel` class. You must pass a valid path to a model YAML file when creating the object. 

2. Instantiate an instance of `OptimizationProgram` by using the `utils.optimization.OptimizationProgram` class. This class takes an `SDEModel` and creates the optimization program. Once instantiated, the object can be used to generate all variables and constraints required for the semidefinite program using the following member functions:
    - `create_opt_vars()` - Generate optimization variables
    - `create_mm_constraints()` - Create and return all moment matrix constraints
    - `create_lm_constraints()` - Create and return all localizing matrix constraints
    - `create_lm_eq_constraints()` - Create and return all localizing matrix scalar equality constraints 
    - `create_mt_constraints()` - Create and return linear evolution (martingale) constriants

3. Generate the optimization objective with `cvxpy` and create the problem object using `cvxpy.Problem` and constraints from step 2. 

For more details, refer to the `example_runner.py` script included in the repository. You may also run the script and see the optimization run for a Brownian motion example by using the following command:

```
python example_runner.py
```

The output will be stored in a text file: `example_output.txt`. 


<!-- ## Citation
If you find the code or the paper useful for your research, please cite our paper:
```
@inproceedings{liu2020decentralized,
  title={Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning},
  author={Liu, Shuijing and Chang, Peixin and Liang, Weihang and Chakraborty, Neeloy and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021},
  pages={3517-3524}
}
``` -->

<!-- ## Credits
Other contributors:  
[Peixin Chang](https://github.com/PeixinC)  
[Neeloy Chakraborty](https://github.com/TheNeeloy)  

Part of the code is based on the following repositories:  

[1] C. Chen, Y. Liu, S. Kreiss, and A. Alahi, “Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning,” in International Conference on Robotics and Automation (ICRA), 2019, pp. 6015–6022.
(Github: https://github.com/vita-epfl/CrowdNav)

[2] I. Kostrikov, “Pytorch implementations of reinforcement learning algorithms,” https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail, 2018.

[3] A. Vemula, K. Muelling, and J. Oh, “Social attention: Modeling attention in human crowds,” in IEEE international Conference on Robotics and Automation (ICRA), 2018, pp. 1–7.
(Github: https://github.com/jeanoh/big)

## Contact
If you have any questions or find any bugs, please feel free to open an issue or pull request. -->