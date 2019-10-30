# experiments core class

import tigercontrol
from tigercontrol.experiments import metrics as metrics_module
from tigercontrol import error
import jax.numpy as np
from tigercontrol.utils.random import set_key
from tqdm import tqdm
import inspect
import time
import operator

metrics = {'mse': metrics_module.mse, 'cross_entropy': metrics_module.cross_entropy}

def to_dict(x):
    '''
    Description: If x is not a dictionary, transforms it to one by assigning None values to entries of x;
                 otherwise, returns x.

    Args:     
        x (dict / list): either a dictionary or a list of keys for the dictionary

    Returns:
        A dictionary 'version' of x
    '''
    if(x is None):
        return {}
    elif(type(x) is not dict):
        x_dict = {}
        for key in x:
            x_dict[key] = [(key, None)]
        return x_dict
    else:
        return x

def get_ids(x):
    '''
    Description: Gets the ids of environments/controllers

    Args:
        x (list / dict): list of ids of environments/controllers or dictionary of environments/controllers and parameters
    Returns:
        x (list): list of environment/controllers ids
    '''
    if(type(x) is dict):
        ids = []
        for main_id in x.keys():
            for (custom_id, _) in x[main_id]:
                ids.append(custom_id)
        return ids
    else:
        return x

def create_full_environment_to_controllers(environments_ids, controller_ids):
    '''
    Description: Associate all given environments to all given controllers.

    Args:
        environment_ids (list): list of environment names
        controller_ids (list): list of controller names
    Returns:
        full_environment_to_controllers (dict): association environment -> controller
    '''
    full_environment_to_controllers = {}

    for environment_id in environments_ids:
        full_environment_to_controllers[environment_id] = []
        for controller_id in controller_ids:
            full_environment_to_controllers[environment_id].append(controller_id)

    return full_environment_to_controllers

def run_experiment(environment, controller, metric = 'mse', key = 0, timesteps = None, verbose = 0):
    '''
    Description: Initializes the experiment instance.
    
    Args:
        environment (tuple): environment id and parameters to initialize the specific environment instance with
        controller (tuple): controller id and parameters to initialize the specific controller instance with
        metric (string): metric we are interesting in computing for current experiment
        key (int): for reproducibility
        timesteps(int): number of time steps to run experiment for
    Returns:
        loss (list): loss series for the specified metric over the entirety of the experiment
        time (float): time elapsed
        memory (float): memory used
    '''
    set_key(key)

    # extract specifications
    (environment_id, environment_params) = environment
    (controller_id, controller_params) = controller
    loss_fn = metrics[metric]

    # initialize environment
    environment = tigercontrol.environment(environment_id)
    if(environment_params is None):
        init = environment.initialize()
    else:
        init = environment.initialize(**environment_params)

    if(timesteps is None):
        if(environment.max_T == -1):
            print("WARNING: On simulated environment, the number of timesteps should be specified. Will default to 10000.")
            timesteps = 10000
        else:
            timesteps = environment.max_T - 1
    elif(environment.max_T != -1):
        if(timesteps > environment.max_T - 1):
            print("WARNING: Number of specified timesteps exceeds the length of the dataset. Will run %d timesteps instead." % environment.max_T - 1)
        timesteps = min(timesteps, environment.max_T - 1)

    # get first x and y
    if(environment.has_regressors):
        x, y = init
    else:
        x, y = init, environment.step()

    # initialize controller
    controller = tigercontrol.controllers(controller_id)
    
    if(controller_params is None):
        controller_params = {}
    if(len(x.shape) == 0):
        controller_params['n'] = 1
    else:
        controller_params['n'] = x.shape[0]
    if(len(y.shape) == 0):
        controller_params['m'] = 1
    else:
        controller_params['m'] = y.shape[0]

    controller.initialize(**controller_params)

    if(verbose):
        print("Running %s on %s..." % (controller_id, environment_id))

    loss = []
    time_start = time.time()
    memory = 0
    load_bar = False
    if(verbose == 2):
        load_bar = True

    # get loss series
    for i in tqdm(range(timesteps), disable = (not load_bar)):
        # get loss and update controller
        cur_loss = float(loss_fn(y, controller.predict(x)))
        loss.append(cur_loss)
        controller.update(y)
        # get new pair of observation and label
        new = environment.step()
        if(environment.has_regressors):
            x, y = new
        else:
            x, y = y, new

    return np.array(loss), time.time() - time_start, memory

def run_experiments(environment, controller, metric = 'mse', n_runs = 1, timesteps = None, verbose = 0):
    
    '''
    Description: Initializes the experiment instance.
    
    Args:
        environment (tuple): environment id and parameters to initialize the specific environment instance with
        controller (tuple): controller id and parameters to initialize the specific controller instance with
        metric (string): metric we are interesting in computing for current experiment
        key (int): for reproducibility
        timesteps(int): number of time steps to run experiment for
    Returns:
        loss (list): loss series for the specified metric over the entirety of the experiment
        time (float): time elapsed
        memory (float): memory used
    '''

    results = tuple((1 / n_runs) * result for result in run_experiment(environment, controller, metric = metric, \
        key = 0, timesteps = timesteps, verbose = verbose))

    for i in range(1, n_runs):
        new_results = tuple((1 / n_runs) * result for result in run_experiment(environment, controller, metric = metric, \
        key = i, timesteps = timesteps, verbose = verbose))
        results = tuple(map(operator.add, results, new_results))

    return results
    