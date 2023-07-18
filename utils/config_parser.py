import yaml


def config_parser(path):
    '''
    Load and parse configuration yaml file.

    Parameters: 
        path (str): path to configuration file
    Returns:
        (dict): dictionary of config settings
    '''
    stream = open(path, 'r')
    config_data = yaml.load(stream, yaml.SafeLoader)

    assert 'solver' in config_data, 'Invalid config file: missing solver name'
    assert 'K' in config_data, 'Invalid config file: maximum moment degree K'
    assert 'mode' in config_data, 'Invalid config file: missing mode'
    assert 'exit_time_moments' in config_data, 'Invalid config file: missing list of exit time moments'
    assert 'model_filename' in config_data, 'Invalid config file: model filename'
    assert 'model_path' in config_data, 'Invalid config file: missing model path'

    solver = config_data['solver']
    K = config_data['K']
    mode = config_data['mode']
    exit_time_moments = config_data['exit_time_moments']
    model_filename = config_data['model_filename']
    model_path = config_data['model_path']
    optional_solver_params = config_data['optional_solver_params']

    assert len(
        exit_time_moments) != 0, 'Invalid config file: list of exit time moments cannot be empty'

    return {'solver': solver,
            'K': K,
            'mode': mode,
            'exit_time_moments': exit_time_moments,
            'model_filename': model_filename,
            'model_path': model_path,
            'optional_solver_params': optional_solver_params}
