# Experiment class

import ctsb
from ctsb import error
from ctsb.experiments.core import run_experiment, create_full_problem_to_models

class NewExperiment(object):
    ''' Description: class for implementing algorithms with enforced modularity '''
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
        
    def initialize(self, problems, models, problem_to_models=None, metrics = 'mse'):
        '''
            Description: Initializes the experiment instance. 

            Args:
                loss_fn (function): function mapping (predict_value, true_value) -> loss
                problem_to_param (dict): map of the form problem_id -> hyperparameters for problem
                model_to_param (dict): map of the form model_id -> hyperparameters for model
                problem_to_models (dict) : map of the form problem_id -> list of model_id. If None, then we assume that the
                user wants to test every model in model_to_params against every problem in problem_to_params
        '''
        self.intialized = True
        self.problems, self.models = self.to_dict(problems), self.to_dict(models)
        self.metrics = metrics

        if(problem_to_models is None):
            self.problem_to_models = create_full_problem_to_models(self.problems.keys(), self.models.keys())
        else:
            self.problem_to_models = problem_to_models

        self.prob_model_to_result = {} # map of the form [metric][problem][model] -> loss series / time / memory

    def run_all_experiments(self, timesteps):
        '''
        Descripton: Runs all experiments for specified number of timesteps.
        
        Args:
            time_steps (int): number of time steps 
        '''
        for metric in self.metrics:
            for problem_id, problem_params in self.problems.keys():
                self.prob_model_to_result[metric][problem] = {}
                for model_id in self.problem_to_models[problem_id]:
                    model_params = self.models[model_id]
                    loss, time, memory = run_experiment((problem_id, problem_params), (model_id, model_params), metric, timesteps = timesteps)
                    self.prob_model_to_result[metric][problem_id][model_id] = loss
                    self.prob_model_to_result['time'][problem_id][model_id] = time
                    self.prob_model_to_result['memory'][problem_id][model_id] = memory    

    def compute_prob_model_to_loss(self, timesteps = 100):
        self.run_all_experiments(timesteps)
        return self.prob_model_to_results

    def help(self):
        # prints information about this class and its methods
        raise NotImplementedError
