# NewExperiment class

import ctsb
from ctsb import error
from ctsb.experiments.core import run_experiment, create_full_problem_to_models

class NewExperiment(object):

    def __init__(self):
        self.initialized = False

    def to_dict(self, x):
        '''
        Description: If x is not a dictionary, transforms it to one by assigning None values to entries of x;
                     otherwise, returns x.

        Args:     
            x (dict / list): either a dictionary or a list of keys for the dictionary

        Returns:
            A dictionary 'version' of x
        '''
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
        Description: Initializes the new experiment instance. 

        Args:     
            problems (dict/list): map of the form problem_id -> hyperparameters for problem or list of problem ids;
                                  in the latter case, default parameters will be used for initialization
            models (dict/list): map of the form model_id -> hyperparameters for model or list of model ids;
                                in the latter case, default parameters will be used for initialization
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
        Descripton: Runs all experiments and returns results

        Args:
            None

        Returns:
            prob_model_to_result (dict): Dictionary containing results for all specified metrics and performance
                                         (time and memory usage) for all problem-model associations.
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
        '''
        Description: Prints information about this class and its methods.
        '''
        print(NewExperiment_help)

    def __str__(self):
        return "<NewExperiment Model>"

# string to print when calling help() method
NewExperiment_help = """

-------------------- *** --------------------

Methods:

    initialize()
        Description: Initializes the new experiment instance. 

        Args:     
            problems (dict/list): map of the form problem_id -> hyperparameters for problem or list of problem ids;
                                  in the latter case, default parameters will be used for initialization
            models (dict/list): map of the form model_id -> hyperparameters for model or list of model ids;
                                in the latter case, default parameters will be used for initialization
            problem_to_models (dict) : map of the form problem_id -> list of model_id.
                                       If None, then we assume that the user wants to
                                       test every model in model_to_params against every
                                       problem in problem_to_params

    def run_all_experiments():
        Descripton: Runs all experiments and returns results

        Args:
            None

        Returns:
            prob_model_to_result (dict): Dictionary containing results for all specified metrics and performance
                                         (time and memory usage) for all problem-model associations.


    help()
        Description: Prints information about this class and its methods

        Args:
            None

        Returns:
            None

-------------------- *** --------------------

"""