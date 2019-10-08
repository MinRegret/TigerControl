# experiments core class

import tigercontrol
from tigercontrol.experiments import metrics as metrics_module
from tigercontrol import error
import jax.numpy as np
from tigercontrol.problems.time_series import TimeSeriesProblem
from tigercontrol.methods.time_series import TimeSeriesMethod
from tigercontrol.utils.random import set_key
from tqdm import tqdm
import inspect
import time

############## TO MAKE AUTOMATIC !!! #################
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
    Description: Gets the ids of problems/methods

    Args:
        x (list / dict): list of ids of problems/methods or dictionary of problems/methods and parameters
    Returns:
        x (list): list of problem/methods ids
    '''
    if(type(x) is dict):
        ids = []
        for main_id in x.keys():
            for (custom_id, _) in x[main_id]:
                ids.append(custom_id)
        return ids
    else:
        return x

def create_full_problem_to_methods(problems_ids, method_ids):
    '''
    Description: Associate all given problems to all given methods.

    Args:
        problem_ids (list): list of problem names
        method_ids (list): list of method names
    Returns:
        full_problem_to_methods (dict): association problem -> method
    '''
    full_problem_to_methods = {}

    for problem_id in problems_ids:
        full_problem_to_methods[problem_id] = []
        for method_id in method_ids:
            full_problem_to_methods[problem_id].append(method_id)

    return full_problem_to_methods

##### CURRENTLY ONLY WORKS WITH TIME SERIES #######
def run_experiment(problem, method, metric = 'mse', key = 0, timesteps = 100, verbose = True, load_bar = True):
    '''
    Description: Initializes the experiment instance.
    
    Args:
        problem (tuple): problem id and parameters to initialize the specific problem instance with
        method (tuple): method id and parameters to initialize the specific method instance with
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
    (problem_id, problem_params) = problem
    (method_id, method_params) = method
    loss_fn = metrics[metric]


    # initialize problem
    problem = tigercontrol.problem(problem_id)
    if(problem_params is None):
        init = problem.initialize()
    else:
        init = problem.initialize(**problem_params)

    # get first x and y
    if(problem.has_regressors):
        x, y = init
    else:
        x, y = init, problem.step()

    # initialize method
    method = tigercontrol.method(method_id)
    if(method_params is None):
        method.initialize()
    else:
        method.initialize(**method_params)

    '''if(problem.has_regressors and not method.uses_regressors):
                    print("ERROR: %s has regressors but %s only uses output signal." % (problem_id, method_id))
                    return np.zeros(timesteps), 0.0, 0.0'''

    if(method.compatibles.isdisjoint(problem.compatibles)): 
        print("ERROR: %s and %s are incompatible!" % (problem_id, method_id))
        return np.zeros(timesteps), 0.0, 0.0

    if(verbose):
        print("Running %s on %s..." % (method_id, problem_id))

    loss = []
    time_start = time.time()
    memory = 0

    # get loss series
    for i in tqdm(range(timesteps), disable = (not load_bar)):
        # get loss and update method
        cur_loss = float(loss_fn(y, method.predict(x)))
        loss.append(cur_loss)
        method.update(y)
        # get new pair of observation and label
        new = problem.step()
        if(problem.has_regressors):
            x, y = new
        else:
            x, y = y, new

    return np.array(loss), time.time() - time_start, memory

    