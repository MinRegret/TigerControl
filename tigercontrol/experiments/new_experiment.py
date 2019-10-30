# NewExperiment class

from tigercontrol import error
from tigercontrol.experiments.core import to_dict, run_experiments, create_full_environment_to_methods

class NewExperiment(object):
    ''' Description: class for implementing algorithms with enforced modularity '''
    def __init__(self):
        self.initialized = False
        
    def initialize(self, environments, methods, environment_to_methods=None, metrics='mse', \
                n_runs = 1, timesteps = None, verbose = 0):
        '''
        Description: Initializes the new experiment instance. 

        Args:     
            environments (dict): map of the form environment_id -> hyperparameters for environment 
            methods (dict): map of the form method_id -> hyperparameters for method
            environment_to_methods (dict) : map of the form environment_id -> list of method_id.
                                       If None, then we assume that the user wants to
                                       test every method in method_to_params against every
                                       environment in environment_to_params
        '''
        self.intialized = True
        self.environments, self.methods, self.metrics = environments, methods, metrics
        self.n_runs, self.timesteps, self.verbose = n_runs, timesteps, verbose

        if(environment_to_methods is None):
            self.environment_to_methods = create_full_environment_to_methods(self.environments.keys(), self.methods.keys())
        else:
            self.environment_to_methods = environment_to_methods

    def run_all_experiments(self):
        '''
        Descripton: Runs all experiments and returns results

        Args:
            None

        Returns:
            prob_method_to_result (dict): Dictionary containing results for all specified metrics and performance
                                         (time and memory usage) for all environment-method associations.
        '''
        prob_method_to_result = {}
        for metric in self.metrics:
            for environment_id in self.environments.keys():
                for (new_environment_id, environment_params) in self.environments[environment_id]:
                    for method_id in self.environment_to_methods[environment_id]:
                        for (new_method_id, method_params) in self.methods[method_id]:
                            loss, time, memory = run_experiments((environment_id, environment_params), (method_id, method_params), \
                                metric, n_runs = self.n_runs, timesteps = self.timesteps, verbose = self.verbose)
                            prob_method_to_result[(metric, environment_id, method_id)] = loss
                            prob_method_to_result[('time', environment_id, method_id)] = time
                            prob_method_to_result[('memory', environment_id, method_id)] = memory

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
            environments (dict): map of the form environment_id -> hyperparameters for environment 
            methods (dict): map of the form method_id -> hyperparameters for method
            environment_to_methods (dict) : map of the form environment_id -> list of method_id.
                                       If None, then we assume that the user wants to
                                       test every method in method_to_params against every
                                       environment in environment_to_params

    def run_all_experiments():
        Descripton: Runs all experiments and returns results

        Args:
            None

        Returns:
            prob_method_to_result (dict): Dictionary containing results for all specified metrics and performance
                                         (time and memory usage) for all environment-method associations.


    help()
        Description: Prints information about this class and its methods

-------------------- *** --------------------

"""