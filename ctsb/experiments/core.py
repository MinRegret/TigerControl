# experiments core class

import ctsb
from ctsb.experiments import metrics as metrics_module
from ctsb import error
import jax.numpy as np
from ctsb.problems.time_series import TimeSeriesProblem
from ctsb.models.time_series import TimeSeriesModel
from ctsb.utils.random import set_key
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
    Description: Gets the ids of problems/models

    Args:
        x (list / dict): list of ids of problems/models or dictionary of problems/models and parameters
    Returns:
        x (list): list of problem/models ids
    '''
    if(type(x) is dict):
        ids = []
        for main_id in x.keys():
            for (custom_id, _) in x[main_id]:
                ids.append(custom_id)
        return ids
    else:
        return x

def create_full_problem_to_models(problems_ids, model_ids):
    '''
    Description: Associate all given problems to all given models.

    Args:
        problem_ids (list): list of problem names
        model_ids (list): list of model names
    Returns:
        full_problem_to_models (dict): association problem -> model
    '''
    full_problem_to_models = {}

    for problem_id in problems_ids:
        full_problem_to_models[problem_id] = []
        for model_id in model_ids:
            full_problem_to_models[problem_id].append(model_id)

    return full_problem_to_models

##### CURRENTLY ONLY WORKS WITH TIME SERIES #######
def run_experiment(problem, model, metric = 'mse', key = 0, timesteps = 100, verbose = True, load_bar = True):
    '''
    Description: Initializes the experiment instance.
    
    Args:
        problem (tuple): problem id and parameters to initialize the specific problem instance with
        model (tuple): model id and parameters to initialize the specific model instance with
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
    (model_id, model_params) = model
    loss_fn = metrics[metric]


    # initialize problem
    problem = ctsb.problem(problem_id)
    if(problem_params is None):
        init = problem.initialize()
    else:
        init = problem.initialize(**problem_params)

    # get first x and y
    if(problem.has_regressors):
        x, y = init
    else:
        x, y = init, problem.step()

    # initialize model
    model = ctsb.model(model_id)
    if(model_params is None):
        model.initialize()
    else:
        model.initialize(**model_params)

    '''if(problem.has_regressors and not model.uses_regressors):
                    print("ERROR: %s has regressors but %s only uses output signal." % (problem_id, model_id))
                    return np.zeros(timesteps), 0.0, 0.0'''

    if(model.compatibles.isdisjoint(problem.compatibles)): 
        print("ERROR: %s and %s are incompatible!" % (problem_id, model_id))
        return np.zeros(timesteps), 0.0, 0.0

    if(verbose):
        print("Running %s on %s..." % (model_id, problem_id))

    loss = []
    time_start = time.time()
    memory = 0

    # get loss series
    for i in tqdm(range(timesteps), disable = (not load_bar)):
        # get loss and update model
        cur_loss = float(loss_fn(y, model.predict(x)))
        loss.append(cur_loss)
        model.update(y)
        # get new pair of observation and label
        new = problem.step()
        if(problem.has_regressors):
            x, y = new
        else:
            x, y = y, new

    return np.array(loss), time.time() - time_start, memory

    