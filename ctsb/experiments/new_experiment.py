# New Experiment class

import ctsb
from ctsb import error
from ctsb.experiments.core import run_experiment, create_full_problem_to_models

# class for implementing algorithms with enforced modularity
class NewExperiment(object):

    def __init__(self):
        self.initialized = False

    def to_dict(self, x):
        if(type(x) is not dict):
            x_dict = {}
            for key in x:
                x_dict[key] = None
            return x_dict
        else:
            return x
        
    def initialize(self, problems, models, problem_to_models=None, metrics = 'mse', timesteps = 1000,\
                         verbose = True, load_bar = True):
        '''
            Description: Initializes the experiment instance. 

            Args:
                loss_fn (function): function mapping (predict_value, true_value) -> loss
                problem_to_param (dict): map of the form problem_id -> hyperparameters for problem
                model_to_param (dict): map of the form model_id -> hyperparameters for model
                problem_to_models (dict) : map of the form problem_id -> list of model_id.
                                           If None, then we assume that the user wants to
                                           test every model in model_to_params against every
                                           problem in problem_to_params
        '''
        self.intialized = True
        self.problems, self.models = self.to_dict(problems), self.to_dict(models)
        self.metrics, self.timesteps, self.verbose, self.load_bar = metrics, timesteps, verbose, load_bar

        if(problem_to_models is None):
            self.problem_to_models = create_full_problem_to_models(self.problems.keys(), self.models.keys())
        else:
            self.problem_to_models = problem_to_models

    def run_all_experiments(self):
        '''
        Descripton:
            Runs all experiments for specified number of timesteps.
        Args:
            time_steps (int): number of time steps 
        '''
        prob_model_to_result = {}
        for metric in self.metrics:
            for problem_id, problem_params in self.problems.items():
                for model_id in self.problem_to_models[problem_id]:
                    model_params = self.models[model_id]
                    loss, time, memory = run_experiment((problem_id, problem_params), (model_id, model_params), metric, timesteps = self.timesteps, verbose = self.verbose, load_bar = self.load_bar)
                    prob_model_to_result[(metric, problem_id, model_id)] = loss
                    prob_model_to_result[('time', problem_id, model_id)] = time
                    prob_model_to_result[('memory', problem_id, model_id)] = memory

        return prob_model_to_result

    def help(self):
        # prints information about this class and its methods
        raise NotImplementedError
