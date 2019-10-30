# NewExperiment class

from tigercontrol import error
from tigercontrol.experiments.core import to_dict, run_experiments, create_full_environment_to_controllers

class NewExperiment(object):
    ''' Description: class for implementing algorithms with enforced modularity '''
    def __init__(self):
        self.initialized = False
        
    def initialize(self, environments, controllers, environment_to_controllers=None, metrics='mse', \
                n_runs = 1, timesteps = None, verbose = 0):
        '''
        Description: Initializes the new experiment instance. 

        Args:     
            environments (dict): map of the form environment_id -> hyperparameters for environment 
            controllers (dict): map of the form controller_id -> hyperparameters for controller
            environment_to_controllers (dict) : map of the form environment_id -> list of controller_id.
                                       If None, then we assume that the user wants to
                                       test every controller in controller_to_params against every
                                       environment in environment_to_params
        '''
        self.intialized = True
        self.environments, self.controllers, self.metrics = environments, controllers, metrics
        self.n_runs, self.timesteps, self.verbose = n_runs, timesteps, verbose

        if(environment_to_controllers is None):
            self.environment_to_controllers = create_full_environment_to_controllers(self.environments.keys(), self.controllers.keys())
        else:
            self.environment_to_controllers = environment_to_controllers

    def run_all_experiments(self):
        '''
        Descripton: Runs all experiments and returns results

        Args:
            None

        Returns:
            prob_controller_to_result (dict): Dictionary containing results for all specified metrics and performance
                                         (time and memory usage) for all environment-controller associations.
        '''
        prob_controller_to_result = {}
        for metric in self.metrics:
            for environment_id in self.environments.keys():
                for (new_environment_id, environment_params) in self.environments[environment_id]:
                    for controller_id in self.environment_to_controllers[environment_id]:
                        for (new_controller_id, controller_params) in self.controllers[controller_id]:
                            loss, time, memory = run_experiments((environment_id, environment_params), (controller_id, controller_params), \
                                metric, n_runs = self.n_runs, timesteps = self.timesteps, verbose = self.verbose)
                            prob_controller_to_result[(metric, environment_id, controller_id)] = loss
                            prob_controller_to_result[('time', environment_id, controller_id)] = time
                            prob_controller_to_result[('memory', environment_id, controller_id)] = memory

        return prob_controller_to_result

    def help(self):
        '''
        Description: Prints information about this class and its controllers.
        '''
        print(NewExperiment_help)

    def __str__(self):
        return "<NewExperiment Controller>"

# string to print when calling help() controller
NewExperiment_help = """

-------------------- *** --------------------

Controllers:

    initialize()
        Description: Initializes the new experiment instance. 

        Args:     
            environments (dict): map of the form environment_id -> hyperparameters for environment 
            controllers (dict): map of the form controller_id -> hyperparameters for controller
            environment_to_controllers (dict) : map of the form environment_id -> list of controller_id.
                                       If None, then we assume that the user wants to
                                       test every controller in controller_to_params against every
                                       environment in environment_to_params

    def run_all_experiments():
        Descripton: Runs all experiments and returns results

        Args:
            None

        Returns:
            prob_controller_to_result (dict): Dictionary containing results for all specified metrics and performance
                                         (time and memory usage) for all environment-controller associations.


    help()
        Description: Prints information about this class and its controllers

-------------------- *** --------------------

"""