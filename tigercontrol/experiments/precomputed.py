''' Precompute '''

from tigercontrol.utils.random import set_key
from tigercontrol.experiments.core import run_experiments, create_full_problem_to_methods
from tigercontrol.utils import get_tigercontrol_dir
import jax.numpy as np
import os
import csv

''' List of all problems and methods '''
all_metrics = ['mse']

all_problems = ['ARMA-v0', 'Crypto-v0', 'SP500-v0']
#
all_methods = ['LastValue', 'AutoRegressor', 'RNN', 'LSTM']

''' Fix timesteps and key '''
timesteps = 1500
n_runs = 10


''' Functions '''
def get_timesteps():
    '''
    Description: Returns number of timesteps used when obtaining precomputed results.
    Args:
        None
    Returns:
        Number of timesteps used for obtaining precomputed results
    '''
    return timesteps

def get_n_runs():
    '''
    Description: Returns key used when obtaining precomputed results.
    Args:
        None
    Returns:
        Number of runs used for obtaining average precomputed results
    '''
    return n_runs

def recompute(verbose = False, load_bar = False):
    '''
    Description: Recomputes all the results.

    Args:
        verbose (boolean): Specifies whether to print what experiment is currently running.
        load_bar (boolean): Specifies whether to show a loading bar while the experiments are running.
    '''

    ''' Store loss series first '''
    for metric in all_metrics:
        for problem_id in all_problems:
            # datapath for current metric and problem
            tigercontrol_dir = get_tigercontrol_dir()
            datapath = 'data/precomputed_results/' + metric + '_' + problem_id[:-3] + '.csv'
            datapath = os.path.join(tigercontrol_dir, datapath)

            with open(datapath, 'w') as csvfile:
                writer = csv.writer(csvfile)
                for method_id in all_methods:
                    try:
                        loss, _, _ = run_experiments((problem_id, None), (method_id, None), metric, \
                                                             n_runs = n_runs, timesteps = timesteps)
                    except:
                        loss = np.zeros(timesteps)
                    # save results for current problem #
                    writer.writerow(loss)
            csvfile.close()

    ''' Store time and memory usage '''
    for problem_id in all_problems:
        # datapath for current metric and problem
        tigercontrol_dir = get_tigercontrol_dir()
        datapath = 'data/precomputed_results/time_memory' + '_' + problem_id[:-3] + '.csv'
        datapath = os.path.join(tigercontrol_dir, datapath)

        with open(datapath, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for method_id in all_methods:
                try:
                    _, time, memory = run_experiments((problem_id, None), (method_id, None), \
                        metric, n_runs = n_runs, timesteps = timesteps, verbose = verbose, load_bar = load_bar)
                except:
                    time, memory = 0.0, 0.0
                # save results for current problem #
                writer.writerow([time, memory])
        csvfile.close()

    print("SUCCESS: EVERYTHING HAS BEEN RECOMPUTED!")

def load_prob_method_to_result(problem_ids = all_problems, method_ids = all_methods, problem_to_methods = None, metrics = 'mse'):
    '''
    Description: Initializes the experiment instance. 

    Args:
        problem_ids (list): ids of problems to evaluate on
        method_ids (list): ids of methods to use
        problem_to_methods (dict): map of the form problem_id -> list of method_id. If None,
                                  then we assume that the user wants to test every method
                                  in method_to_params against every problem in problem_to_params
        metrics (list): metrics to load

     Returns:
        prob_method_to_result (dict): Dictionary containing results for all specified metrics and
                                     performance (time and memory usage) for all problem-method
                                     associations.
    '''

    if(problem_to_methods is None):
        problem_to_methods = create_full_problem_to_methods(problem_ids, method_ids)

    prob_method_to_result = {}

    ''' Get loss series '''
    for metric in metrics:
        for problem_id in problem_ids:
            # datapath for current metric and problem
            tigercontrol_dir = get_tigercontrol_dir()
            datapath = 'data/precomputed_results/' + metric + '_' + problem_id[:-3] + '.csv'
            datapath = os.path.join(tigercontrol_dir, datapath)

            with open(datapath) as csvfile:
                reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                method_no = 0
                for row in reader:
                    if(all_methods[method_no] in method_ids):
                        prob_method_to_result[(metric, problem_id, all_methods[method_no])] = np.array(row)
                    method_no += 1
            csvfile.close()

    ''' Get time and memory usage '''
    for problem_id in problem_ids:
        # datapath for current metric and problem
        tigercontrol_dir = get_tigercontrol_dir()
        datapath = 'data/precomputed_results/time_memory' + '_' + problem_id[:-3] + '.csv'
        datapath = os.path.join(tigercontrol_dir, datapath)

        with open(datapath) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            method_no = 0
            for row in reader:
                if(all_methods[method_no] in method_ids):
                    prob_method_to_result[('time', problem_id, all_methods[method_no])] = row[0]
                    prob_method_to_result[('memory', problem_id, all_methods[method_no])] = row[1]
                method_no += 1
        csvfile.close()

    return prob_method_to_result

def hyperparameter_warning():
    print("WARNING: when using precomputed results, any specified problem hyperparameters" + \
                " will be disregarded and default ones will be used instead.")

if __name__ == "__main__":
    recompute()


