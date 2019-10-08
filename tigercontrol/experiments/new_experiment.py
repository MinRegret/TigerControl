# NewExperiment class

from tigercontrol import error
from tigercontrol.experiments.core import to_dict, run_experiment, create_full_problem_to_methods

class NewExperiment(object):
    ''' Description: class for implementing algorithms with enforced modularity '''
    def __init__(self):
        self.initialized = False
        
    def initialize(self, problems, methods, problem_to_methods=None, metrics = 'mse', key = 0, timesteps = 1000, \
                         verbose = True, load_bar = True):
        '''
        Description: Initializes the new experiment instance. 

        Args:     
            problems (dict): map of the form problem_id -> hyperparameters for problem 
            methods (dict): map of the form method_id -> hyperparameters for method
            problem_to_methods (dict) : map of the form problem_id -> list of method_id.
                                       If None, then we assume that the user wants to
                                       test every method in method_to_params against every
                                       problem in problem_to_params
        '''
        self.intialized = True
        self.problems, self.methods, self.metrics = problems, methods, metrics
        self.key, self.timesteps, self.verbose, self.load_bar = key, timesteps, verbose, load_bar

        if(problem_to_methods is None):
            self.problem_to_methods = create_full_problem_to_methods(self.problems.keys(), self.methods.keys())
        else:
            self.problem_to_methods = problem_to_methods

    def run_all_experiments(self):
        '''
        Descripton: Runs all experiments and returns results

        Args:
            None

        Returns:
            prob_method_to_result (dict): Dictionary containing results for all specified metrics and performance
                                         (time and memory usage) for all problem-method associations.
        '''
        prob_method_to_result = {}
        for metric in self.metrics:
            for problem_id in self.problems.keys():
                for (new_problem_id, problem_params) in self.problems[problem_id]:
                    for method_id in self.problem_to_methods[problem_id]:
                        for (new_method_id, method_params) in self.methods[method_id]:
                            loss, time, memory = run_experiment((problem_id, problem_params), (method_id, method_params), metric, key = self.key, timesteps = self.timesteps, verbose = self.verbose, load_bar = self.load_bar)
                            prob_method_to_result[(metric, problem_id, method_id)] = loss
                            prob_method_to_result[('time', problem_id, method_id)] = time
                            prob_method_to_result[('memory', problem_id, method_id)] = memory

        return prob_method_to_result

    def help(self):
        '''
        Description: Prints information about this class and its methods.
        '''
        print(NewExperiment_help)

    def __str__(self):
        return "<NewExperiment Method>"

# string to print when calling help() method
NewExperiment_help = """

-------------------- *** --------------------

Methods:

    initialize()
        Description: Initializes the new experiment instance. 

        Args:     
            problems (dict): map of the form problem_id -> hyperparameters for problem 
            methods (dict): map of the form method_id -> hyperparameters for method
            problem_to_methods (dict) : map of the form problem_id -> list of method_id.
                                       If None, then we assume that the user wants to
                                       test every method in method_to_params against every
                                       problem in problem_to_params

    def run_all_experiments():
        Descripton: Runs all experiments and returns results

        Args:
            None

        Returns:
            prob_method_to_result (dict): Dictionary containing results for all specified metrics and performance
                                         (time and memory usage) for all problem-method associations.


    help()
        Description: Prints information about this class and its methods

-------------------- *** --------------------

"""