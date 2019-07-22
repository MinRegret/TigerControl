''' Precompute '''

from ctsb.utils.random import set_key
from ctsb.experiments.core import run_experiment, create_full_problem_to_models
from ctsb.utils.download_tools import get_ctsb_dir
import jax.numpy as np
import os
import csv

''' List of all problems and models '''
all_metrics = ['mse', 'cross_entropy']

## no uciindoor-v0 , ctrl indices has some problems##
all_problems = ['Random-v0', 'ARMA-v0', 'SP500-v0', 'Crypto-v0', 'ENSO-v0']

####### LSTM AND RNN NOT INCLUDED BECAUSE THEY NEED INPUT SHAPE ###############
####### and ArmaAdaGrad is not classified as timeseries & other problems ... ###########
all_models = ['LastValue', 'AutoRegressor', 'PredictZero', 'ArmaOgd']

####### NEED TO MAKE IT HARD TO CHANGE !!!!!! ########
''' Fix timesteps and key '''
timesteps = 100
key = 0

''' Ensure repeatability '''
set_key(key)

''' Functions '''
def get_timesteps():
    return timesteps

def get_key():
    return keys

def recompute():
    '''
    Description: Initializes the experiment instance. 
    Args:
    '''

    ''' Store loss series first '''
    for metric in all_metrics:
        for problem_id in all_problems:
            # datapath for current metric and problem
            ctsb_dir = get_ctsb_dir()
            datapath = 'data/precomputed_results/' + metric + '_' + problem_id[:-3] + '.csv'
            datapath = os.path.join(ctsb_dir, datapath)

            with open(datapath, 'w') as csvfile:
                writer = csv.writer(csvfile)
                for model_id in all_models:
                    print("Getting %s: Running %s on %s" % (metric, problem_id[:-3], model_id))
                    #loss, time, memory = run_experiment((problem_id, None), (model_id, None), metric, key = key, timesteps = timesteps)
                    try:
                        loss, time, memory = run_experiment((problem_id, None), (model_id, None), metric, key = key, timesteps = timesteps)
                    except:
                        loss = np.zeros(timesteps)
                    # save results for current problem #
                    writer.writerow(loss)
            csvfile.close()

    ''' Store time and memory usage '''
    for problem_id in all_problems:
        # datapath for current metric and problem
        ctsb_dir = get_ctsb_dir()
        datapath = 'data/precomputed_results/time_memory' + '_' + problem_id[:-3] + '.csv'
        datapath = os.path.join(ctsb_dir, datapath)

        with open(datapath, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for model_id in all_models:
                print("Getting Performance: Running %s on %s" % (problem_id[:-3], model_id))
                try:
                    _, time, memory = run_experiment((problem_id, None), (model_id, None), metric, key = key, timesteps = timesteps)
                except:
                    time, memory = -1, -1
                # save results for current problem #
                writer.writerow([time, memory])
        csvfile.close()

    print("SUCCESS: EVERYTHING HAS BEEN RECOMPUTED!")

def load_prob_model_to_result(problem_ids = all_problems, model_ids = all_models, problem_to_models = None, metrics = 'mse'):
    '''
    Description:
        Initializes the experiment instance. 
    Args:
    '''

    if(problem_to_models is None):
        problem_to_models = create_full_problem_to_models(problem_ids, model_ids)

    prob_model_to_result = {}

    ''' Get loss series '''
    for metric in metrics:
        for problem_id in problem_ids:
            # datapath for current metric and problem
            ctsb_dir = get_ctsb_dir()
            datapath = 'data/precomputed_results/' + metric + '_' + problem_id[:-3] + '.csv'
            datapath = os.path.join(ctsb_dir, datapath)

            with open(datapath) as csvfile:
                reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                model_no = 0
                for row in reader:
                    if(all_models[model_no] in model_ids):
                        prob_model_to_result[(metric, problem_id, all_models[model_no])] = row
                    model_no += 1
            csvfile.close()

    ''' Get time and memory usage '''
    for problem_id in problem_ids:
        # datapath for current metric and problem
        ctsb_dir = get_ctsb_dir()
        datapath = 'data/precomputed_results/time_memory' + '_' + problem_id[:-3] + '.csv'
        datapath = os.path.join(ctsb_dir, datapath)

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


