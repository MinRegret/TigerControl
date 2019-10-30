# Experiment class

from tigercontrol.experiments.core import run_experiments, get_ids, to_dict
from tigercontrol.experiments.new_experiment import NewExperiment
from tigercontrol.experiments import precomputed
import csv
import jax.numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class Experiment(object):
    ''' Description: Streamlines the process of performing experiments and comparing results of methods across
             a range of environments. '''
    def __init__(self):
        self.initialized = False
        
    def initialize(self, environments = None, methods = None, environment_to_methods = None, metrics = ['mse'], \
                   n_runs = 1, use_precomputed = False, timesteps = None, verbose = 0):
        '''
        Description: Initializes the experiment instance. 

        Args:
            environments (dict/list): map of the form environment_id -> hyperparameters for environment or list of environment ids;
                                  in the latter case, default parameters will be used for initialization
            methods (dict/list): map of the form method_id -> hyperparameters for method or list of method ids;
                                in the latter case, default parameters will be used for initialization
            environment_to_methods (dict) : map of the form environment_id -> list of method_id.
                                       If None, then we assume that the user wants to
                                       test every method in method_to_params against every
                                       environment in environment_to_params
            metrics (list): Specifies metrics we are interested in evaluating.
            n_runs (int): Specifies the number of experiments to average over.
            use_precomputed (boolean): Specifies whether to use precomputed results.
            timesteps (int): Number of time steps to run experiment for
            verbose (0, 1, 2): Specifies the verbosity of the experiment instance.
        '''

        self.environments, self.methods = to_dict(environments), to_dict(methods)
        self.environment_to_methods, self.metrics = environment_to_methods, metrics
        self.n_runs, self.use_precomputed = n_runs, use_precomputed
        self.timesteps, self.verbose = timesteps, verbose

        self.n_environments, self.n_methods = {}, {}

        if(use_precomputed):

            if(timesteps > precomputed.get_timesteps()):
                print("WARNING: when using precomputed results, the maximum number of timesteps is fixed. " + \
                    "Will use %d instead of the specified %d" % (precomputed.get_timesteps(), timesteps))
                self.timesteps = precomputed.get_timesteps()

            # ensure environments and methods don't have specified hyperparameters
            if(type(environments) is dict or type(methods) is dict):
                precomputed.hyperparameter_warning()

            # map of the form [metric][environment][method] -> loss series + time + memory
            self.prob_method_to_result = precomputed.load_prob_method_to_result(\
                environment_ids = list(self.environments.keys()), method_ids = list(self.methods.keys()), \
                environment_to_methods = environment_to_methods, metrics = metrics)

        else:
            self.new_experiment = NewExperiment()
            self.new_experiment.initialize(self.environments, self.methods, environment_to_methods, \
                metrics, n_runs, timesteps, verbose)
            # map of the form [metric][environment][method] -> loss series + time + memory
            self.prob_method_to_result = self.new_experiment.run_all_experiments()

    def add_method(self, method_id, method_params = None, name = None):
        '''
        Description: Add a new method to the experiment instance.
        
        Args:
            method_id (string): ID of new method.
            method_params (dict): Parameters to use for initialization of new method.
        '''
        assert method_id is not None, "ERROR: No Method ID given."

        if name is None and 'optimizer' in method_params:
            name = method_params['optimizer'].__name__

        new_id = ''
        if(method_id in self.methods):
            if(method_id not in self.n_methods):
                self.n_methods[method_id] = 0
            self.n_methods[method_id] += 1
            if(name is not None):
                new_id = method_id + '-' + name
            else:
                new_id = method_id + '-' + str(self.n_methods[method_id])
            self.methods[method_id].append((new_id, method_params))
        else:
            new_id = method_id
            if(name is not None):
                new_id += '-' + name
            self.methods[method_id] = [(new_id, method_params)]
            self.n_methods[method_id] = 1

        ''' Evaluate performance of new method on all environments '''
        for metric in self.metrics:
            for environment_id in self.environments.keys():
                for (new_environment_id, environment_params) in self.environments[environment_id]:

                    ''' If method is compatible with environment, run experiment and store results. '''
                    try:
                        loss, time, memory = run_experiments((environment_id, environment_params), \
                            (method_id, method_params), metric = metric, n_runs = self.n_runs, \
                            timesteps = self.timesteps, verbose = self.verbose)
                    except:
                        print("ERROR: Could not run %s on %s." % (method_id, environment_id) + \
                            " Please make sure method and environment are compatible.")
                        loss, time, memory = 0, 0.0, 0.0

                    self.prob_method_to_result[(metric, new_environment_id, new_id)] = loss
                    self.prob_method_to_result[('time', new_environment_id, new_id)] = time
                    self.prob_method_to_result[('memory', new_environment_id, new_id)] = memory

    def add_environment(self, environment_id, environment_params = None, name = None):
        '''
        Description: Add a new environment to the experiment instance.
        
        Args:
            environment_id (string): ID of new method.
            environment_params (dict): Parameters to use for initialization of new method.
        '''
        assert environment_id is not None, "ERROR: No Environment ID given."

        new_id = ''

        # AN INSTANCE OF THE PROBLEM ALREADY EXISTS
        if(environment_id in self.environments):
            # COUNT NUMBER OF INSTANCES OF SAME MAIN PROBLEM
            if(environment_id not in self.n_environments):
                self.n_environments[environment_id] = 0
            self.n_environments[environment_id] += 1
            # GET ID OF PROBLEM INSTANCE
            if(name is not None):
                new_id = name
            else:
                new_id = environment_id[:-2] + str(self.n_environments[environment_id])
            self.environments[environment_id].append((new_id, environment_params))
        # NO INSTANCE OF THE PROBLEM EXISTS
        else:
            new_id = environment_id[:-3]
            if(name is not None):
                new_id = name
            self.environments[environment_id] = [(new_id, environment_params)]
            self.n_environments[environment_id] = 1

        ''' Evaluate performance of new method on all environments '''
        for metric in self.metrics:
            for method_id in self.methods.keys():
                for (new_method_id, method_params) in self.methods[method_id]:

                    ''' If method is compatible with environment, run experiment and store results. '''
                    try:
                        loss, time, memory = run_experiments((environment_id, environment_params), \
                            (method_id, method_params), metric = metric, n_runs = self.n_runs, \
                            timesteps = self.timesteps, verbose = self.verbose)
                    except:
                        print("ERROR: Could not run %s on %s. Please make sure method and environment are compatible." % (method_id, environment_id))
                        loss, time, memory = 0.0, 0.0, 0.0

                    self.prob_method_to_result[(metric, new_id, new_method_id)] = loss
                    self.prob_method_to_result[('time', new_id, new_method_id)] = time
                    self.prob_method_to_result[('memory', new_id, new_method_id)] = memory

    def to_csv(self, table_dict, save_as):
        ''' Save to csv file '''
        with open(save_as, 'w') as f:
            for key in table_dict.keys():
                f.write(key)
                for item in table_dict[key]:
                    f.write(",%s" % str(item))
                f.write('\n')

    def scoreboard(self, metric = 'mse', start_time = 0, n_digits = 3, truncate_ids = True, verbose = True, save_as = None):
        '''
        Description: Show a scoreboard for the results of the experiments for specified metric.

        Args:
            save_as (string): If not None, datapath to save results as csv file.
            metric (string): Metric to compare results
            verbose (boolean): Specifies whether to print the description of the scoreboard entries
        '''

        if(self.use_precomputed and metric == 'time' and len(self.n_methods.keys()) > 0):
            print("WARNING: Time comparison between precomputed methods and" + \
                  "any added method may be irrelevant due to hardware differences.")

        if(verbose and metric in self.metrics):
            print("Average " + metric + ":")
        else:
            print(metric + ":")
            
        table = PrettyTable()
        table_dict = {}

        environment_ids = get_ids(self.environments)
        method_ids = get_ids(self.methods)

        table_dict['Environments'] = environment_ids

        field_names = ['Method\Environments']
        for environment_id in environment_ids:
            if(truncate_ids and len(environment_id) > 9):
                field_names.append(environment_id[:4] + '..' + environment_id[-3:])
            else:
                field_names.append(environment_id)

        table.field_names = field_names

        for method_id in method_ids:
            method_scores = [method_id]
            # get scores for each environment
            for environment_id in environment_ids:
                score = np.mean((self.prob_method_to_result\
                    [(metric, environment_id, method_id)])[start_time:self.timesteps])
                score = round(float(score), n_digits)
                if(score == 0.0):
                    score = '—'
                method_scores.append(score)
            table.add_row(method_scores)
            table_dict[method_id] = method_scores[1:]

        print(table)

        if(save_as is not None):
            self.to_csv(table_dict, save_as)

    def avg_regret(self, loss):
        avg_regret = []
        cur_avg = 0
        for i in range(len(loss)):
            cur_avg = (i / (i + 1)) * cur_avg + loss[i] / (i + 1)
            avg_regret.append(cur_avg)
        return avg_regret

    def _plot(self, ax, environment, environment_result_plus_method, n_environments, metric, \
                avg_regret, start_time, cutoffs, yscale, show_legend = True):

        for (loss, method) in environment_result_plus_method:
            if(avg_regret):
                ax.plot(self.avg_regret(loss[start_time:self.timesteps]), label=str(method))
            else:
                ax.plot(loss, label=str(method))
        if(show_legend):
            ax.legend(loc="upper right", fontsize=5 + 5//n_environments)
        ax.set_title("Environment:" + str(environment))
        #ax.set_xlabel("timesteps")
        ax.set_ylabel(metric)

        if(cutoffs is not None and environment in cutoffs.keys()):
            ax.set_ylim([0, cutoffs[environment]])

        if(yscale is not None):
            ax.set_yscale(yscale)

        return ax

    def graph(self, environment_ids = None, metric = 'mse', avg_regret = True, start_time = 0, \
            cutoffs = None, yscale = None, time = 20, save_as = None, size = 3, dpi = 100):

        '''
        Description: Show a graph for the results of the experiments for specified metric.
        
        Args:
            save_as (string): If not None, datapath to save the figure containing the plots
            metric (string): Metric to compare results
            time (float): Specifies how long the graph should display for
        '''

        # check metric exists
        assert metric in self.metrics

        # get environment and method ids
        if(environment_ids is None):
            environment_ids = get_ids(self.environments)
        method_ids = get_ids(self.methods)

        # get number of environments
        n_environments = len(environment_ids)

        all_environment_info = []

        for environment_id in environment_ids:
            environment_result_plus_method = []
            method_list = []
            for method_id in method_ids:
                method_list.append(method_id)
                environment_result_plus_method.append((self.prob_method_to_result[(metric, environment_id, method_id)], method_id))
            all_environment_info.append((environment_id, environment_result_plus_method, method_list))

        nrows = max(int(np.sqrt(n_environments)), 1)
        ncols = n_environments // nrows + n_environments % nrows

        fig, ax = plt.subplots(figsize = (ncols * size, nrows * size), nrows=nrows, ncols=ncols)
        fig.canvas.set_window_title('TigerSeries')

        if n_environments == 1:
            (environment, environment_result_plus_method, method_list) = all_environment_info[0]
            ax = self._plot(ax, environment, environment_result_plus_method, n_environments, \
                metric, avg_regret, start_time, cutoffs, yscale)
        elif nrows == 1:
            for j in range(ncols):
                (environment, environment_result_plus_method, method_list) = all_environment_info[j]
                ax[j] = self._plot(ax[j], environment, environment_result_plus_method, n_environments, \
                                          metric, avg_regret, start_time, cutoffs, yscale)
        else:
            cur_pb = 0
            for i in range(nrows):
                for j in range(ncols):

                    if(cur_pb == n_environments):
                        legend = []
                        for method_id in method_ids:
                            legend.append((0, method_id))
                        ax[i, j] = self._plot(ax[i, j], 'LEGEND', legend,\
                                n_environments, metric, False, cutoffs, None, show_legend = True)
                        continue

                    if(cur_pb > n_environments):
                        ax[i, j].plot(0, 'x', 'red', label="NO MORE \n MODELS")
                        ax[i, j].legend(loc="center", fontsize=8 + 10//n_environments)
                        continue

                    (environment, environment_result_plus_method, method_list) = all_environment_info[cur_pb]
                    cur_pb += 1
                    ax[i, j] = self._plot(ax[i, j], environment, environment_result_plus_method,\
                                n_environments, metric, avg_regret, start_time, cutoffs, yscale, show_legend = False)

        #fig.tight_layout()

        if(save_as is not None):
            plt.savefig(save_as, dpi=dpi)

        if time:
            plt.show(block=False)
            plt.pause(time)
            plt.close()
        else:
            plt.show()

    def help(self):
        '''
        Description: Prints information about this class and its methods.
        '''
        print(Experiment_help)

    def __str__(self):
        return "<Experiment Method>"

# string to print when calling help() method
Experiment_help = """

-------------------- *** --------------------

Description: Streamlines the process of performing experiments and comparing results of methods across
             a range of environments.

Methods:

    initialize(environments = None, methods = None, environment_to_methods = None, metrics = ['mse'],
               use_precomputed = True, timesteps = 100, verbose = True, load_bar = True):

        Description: Initializes the experiment instance. 

        Args:
            environments (dict/list): map of the form environment_id -> hyperparameters for environment or list of environment ids;
                                  in the latter case, default parameters will be used for initialization

            methods (dict/list): map of the form method_id -> hyperparameters for method or list of method ids;
                                in the latter case, default parameters will be used for initialization

            environment_to_methods (dict) : map of the form environment_id -> list of method_id.
                                       If None, then we assume that the user wants to
                                       test every method in method_to_params against every
                                       environment in environment_to_params

            metrics (list): Specifies metrics we are interested in evaluating.

            use_precomputed (boolean): Specifies whether to use precomputed results.

            timesteps (int): Number of time steps to run experiment for

            verbose (boolean): Specifies whether to print what experiment is currently running.

            load_bar (boolean): Specifies whether to show a loading bar while the experiments are running.


    add_method(method_id, method_params = None):

        Description: Add a new method to the experiment instance.
        
        Args:
            method_id (string): ID of new method.

            method_params: Parameters to use for initialization of new method.


    scoreboard(save_as = None, metric = 'mse'):

        Description: Show a scoreboard for the results of the experiments for specified metric.

        Args:
            save_as (string): If not None, datapath to save results as csv file.

            metric (string): Metric to compare results

            verbose (boolean): Specifies whether to print the description of the scoreboard entries


    graph(save_as = None, metric = 'mse', time = 5):

        Description: Show a graph for the results of the experiments for specified metric.

        Args:
            save_as (string): If not None, datapath to save the figure containing the plots

            metric (string): Metric to compare results
            
            time (float): Specifies how long the graph should display for

    help()

        Description: Prints information about this class and its methods

-------------------- *** --------------------

"""
