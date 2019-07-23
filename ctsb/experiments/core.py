# experiments core class

import ctsb
from ctsb.experiments import metrics as metrics_module
from ctsb import error
from ctsb.problems.time_series import TimeSeriesProblem
from ctsb.models.time_series import TimeSeriesModel
from ctsb.utils.random import set_key
from tqdm import tqdm
import inspect
import time

############## TO MAKE AUTOMATIC !!! #################
metrics = {'mse': metrics_module.mse, 'cross_entropy': metrics_module.cross_entropy}

def get_ids(x):
    '''
    Description: Gets the ids of problems/models

    Args:
        x (list / dict): list of ids of problems/models or dictionary of problems/models and parameters
    Returns:
        x (list): list of problem/models ids
    '''
    if(x is dict):
        return list(x.keys())
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
def run_experiment(problem, model, metric = 'mse', key = None, timesteps = 100, verbose = True, load_bar = True):
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
        init = problem.initialize(problem_params)

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
        model.initialize(model_params)

    if(problem.has_regressors and not model.uses_regressors):
        print("WARNING: Problem has regressors but model only uses output signal.")

    # check problem and model are of the same type
    is_ts_problem = (inspect.getmro(problem.__class__))[1] == TimeSeriesProblem
    is_ts_model = (inspect.getmro(model.__class__))[1] == TimeSeriesModel
    # assert (is_ts_problem and is_ts_model), "ERROR: Currently Experiment only supports Time Series Problems and Models."

    if(verbose):
        print("Running %s on %s..." % (model_id, problem_id))

    loss = []
    time_start = time.time()
    memory = 0

    # get loss series
    for i in tqdm(range(timesteps), disable = (not load_bar)):
        # get loss and update model
        cur_loss = loss_fn(y, model.predict(x))
        loss.append(cur_loss)
        model.update(y)
        # get new pair of observation and label
        new = problem.step()
        if(problem.has_regressors):
            x, y = new
        else:
            x, y = y, new

    return loss, time.time() - time_start, memory

    