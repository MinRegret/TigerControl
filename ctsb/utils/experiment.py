# Experiment class
# Author: John Hallman

from ctsb import error
import jax.numpy as np
from tqdm import tqdm

# class for implementing algorithms with enforced modularity
class Experiment(object):
    def __init__(self):
        self.initialized = False
    
    def initialize(self, problem_obs_models_list, loss_fn, timesteps):
        '''
        Args:
            problem_obs_models_list(list): list of tuples of the form (initalized problem, first_obs, [list of initialized models])
            loss_fn(function): function mapping (predict_value, true_value) -> loss
            timesteps(int): total number of steps to run experiment
        Returns:
            Instance of Experiment class
        '''
        self.intialized = True
        self.T = timesteps
        self.loss = loss_fn
        self.pom_ls = problem_obs_models_list
        self.prob_model_to_loss = {} # map of the form [problem][model] -> loss series

    def run_all_experiments(self):
        for (problem, obs, models) in self.pom_ls:
            self.prob_model_to_loss[problem] = {}
            for model in models:
                self.prob_model_to_loss[problem][model] = []
                self.run_experiment(problem, obs, model)
        return

    def run_experiment(self, problem, obs, model):
        cur_x = obs
        print ("running experiment: " + str(model) + " on " + str(problem))
        for i in tqdm(range(0,self.T)):
            cur_y_pred = model.predict(cur_x)
            cur_y_true= problem.step()
            cur_loss = self.loss(cur_y_true, cur_y_pred)
            self.prob_model_to_loss[problem][model].append(cur_loss)
            model.update(cur_loss)
            cur_x = cur_y_true
        return

    def 
    def get_prob_model_to_loss(self):
        return self.prob_model_to_loss

    def help(self):
        # prints information about this class and its methods
        raise NotImplementedError
    '''
    def __str__(self):
        if self.spec is None:
            return '<{} instance> call object help() method for info'.format(type(self).__name__)
        else:
            return '<{}<{}>> call object help() method for info'.format(type(self).__name__, self.spec.id)
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        # propagate exception
        return False
    '''


