# NewExperiment class

from tigercontrol import error
from tigercontrol.experiments.core import to_dict, run_experiment, create_full_problem_to_models

class NewExperiment(object):
    ''' Description: class for implementing algorithms with enforced modularity '''
    def __init__(self):
        self.initialized = False
        
    def initialize(self, problems, models, problem_to_models=None, metrics = 'mse', key = 0, timesteps = 1000, \
                         verbose = True, load_bar = True):
        '''
        Description: Initializes the new experiment instance. 

        Args:     
            problems (dict): map of the form problem_id -> hyperparameters for problem 
            models (dict): map of the form model_id -> hyperparameters for model
            problem_to_models (dict) : map of the form problem_id -> list of model_id.
                                       If None, then we assume that the user wants to
                                       test every model in model_to_params against every
                                       problem in problem_to_params
        '''
        self.intialized = True
        self.problems, self.models, self.metrics = problems, models, metrics
        self.key, self.timesteps, self.verbose, self.load_bar = key, timesteps, verbose, load_bar

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
            for problem_id in self.problems.keys():
                for (new_problem_id, problem_params) in self.problems[problem_id]:
                    for model_id in self.problem_to_models[problem_id]:
                        for (new_model_id, model_params) in self.models[model_id]:
                            loss, time, memory = run_experiment((problem_id, problem_params), (model_id, model_params), metric, key = self.key, timesteps = self.timesteps, verbose = self.verbose, load_bar = self.load_bar)
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
            problems (dict): map of the form problem_id -> hyperparameters for problem 
            models (dict): map of the form model_id -> hyperparameters for model
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

-------------------- *** --------------------

"""