''' Precompute '''

from tigercontrol.utils.random import set_key
from tigercontrol.experiments.core import run_experiment, create_full_problem_to_models
from tigercontrol.utils.download_tools import get_tigercontrol_dir
import jax.numpy as np
import os
import csv

''' List of all problems and models '''
all_metrics = ['mse']

## no uciindoor-v0 , ctrl indices has some problems##
all_problems = ['ARMA-v0', 'Crypto-v0', 'SP500-v0']

####### LSTM AND RNN NOT INCLUDED BECAUSE THEY NEED INPUT SHAPE ###############
####### and ArmaAdaGrad is not classified as timeseries & other problems ... ###########
all_models = ['LastValue', 'AutoRegressor', 'RNN', 'LSTM']

####### NEED TO MAKE IT HARD TO CHANGE !!!!!! ########
''' Fix timesteps and key '''
timesteps = 1500
key = 0

''' Ensure repeatability '''
set_key(key)

''' Functions '''
def get_timesteps():
    '''
    Description: Returns number of timesteps used when obtaining precomputed results.

    Args:
        None

    Returns:
        timesteps used for obtaining precomputed results
    '''
    return timesteps

def get_key():
    '''
    Description: Returns key used when obtaining precomputed results.

    Args:
        None

    Returns:
        key used for obtaining precomputed results
    '''
    return key

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
                for model_id in all_models:
                    try:
                        loss, time, memory = run_experiment((problem_id, None), (model_id, None), metric, \
                                                             key = key, timesteps = timesteps)
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
            for model_id in all_models:
                try:
                    _, time, memory = run_experiment((problem_id, None), (model_id, None), \
                        metric, key = key, timesteps = timesteps, verbose = verbose, load_bar = load_bar)
                except:
                    time, memory = 0.0, 0.0
                # save results for current problem #
                writer.writerow([time, memory])
        csvfile.close()

    print("SUCCESS: EVERYTHING HAS BEEN RECOMPUTED!")

def load_prob_model_to_result(problem_ids = all_problems, model_ids = all_models, problem_to_models = None, metrics = 'mse'):
    '''
    Description: Initializes the experiment instance. 

    Args:
        problem_ids (list): ids of problems to evaluate on
        model_ids (list): ids of models to use
        problem_to_models (dict): map of the form problem_id -> list of model_id. If None,
                                  then we assume that the user wants to test every model
                                  in model_to_params against every problem in problem_to_params
        metrics (list): metrics to load

     Returns:
        prob_model_to_result (dict): Dictionary containing results for all specified metrics and
                                     performance (time and memory usage) for all problem-model
                                     associations.
    '''

    if(problem_to_models is None):
        problem_to_models = create_full_problem_to_models(problem_ids, model_ids)

    prob_model_to_result = {}

    ''' Get loss series '''
    for metric in metrics:
        for problem_id in problem_ids:
            # datapath for current metric and problem
            tigercontrol_dir = get_tigercontrol_dir()
            datapath = 'data/precomputed_results/' + metric + '_' + problem_id[:-3] + '.csv'
            datapath = os.path.join(tigercontrol_dir, datapath)

            with open(datapath) as csvfile:
                reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                model_no = 0
                for row in reader:
                    if(all_models[model_no] in model_ids):
                        prob_model_to_result[(metric, problem_id, all_models[model_no])] = np.array(row)
                    model_no += 1
            csvfile.close()

    ''' Get time and memory usage '''
    for problem_id in problem_ids:
        # datapath for current metric and problem
        tigercontrol_dir = get_tigercontrol_dir()
        datapath = 'data/precomputed_results/time_memory' + '_' + problem_id[:-3] + '.csv'
        datapath = os.path.join(tigercontrol_dir, datapath)

        with open(datapath) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            model_no = 0
            for row in reader:
                if(all_models[model_no] in model_ids):
                    prob_model_to_result[('time', problem_id, all_models[model_no])] = row[0]
                    prob_model_to_result[('memory', problem_id, all_models[model_no])] = row[1]
                model_no += 1
        csvfile.close()

    return prob_model_to_result

if __name__ == "__main__":
    recompute()


