# Experiment class

from tigercontrol import error
from tigercontrol.experiments.core import run_experiment, get_ids, to_dict
from tigercontrol.experiments.new_experiment import NewExperiment
from tigercontrol.experiments import precomputed
import csv
import jax.numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class Experiment(object):
    ''' Description: Experiment class '''
    def __init__(self):
        self.initialized = False
        
    def initialize(self, problems = None, methods = None, problem_to_methods = None, metrics = ['mse'], \
                   key = 0, use_precomputed = False, timesteps = 100, verbose = False, load_bar = False):
        '''
        Description: Initializes the experiment instance. 

        Args:
            problems (dict/list): map of the form problem_id -> hyperparameters for problem or list of problem ids;
                                  in the latter case, default parameters will be used for initialization
            methods (dict/list): map of the form method_id -> hyperparameters for method or list of method ids;
                                in the latter case, default parameters will be used for initialization
            problem_to_methods (dict) : map of the form problem_id -> list of method_id.
                                       If None, then we assume that the user wants to
                                       test every method in method_to_params against every
                                       problem in problem_to_params
            metrics (list): Specifies metrics we are interested in evaluating.
            use_precomputed (boolean): Specifies whether to use precomputed results.
            timesteps (int): Number of time steps to run experiment for
            verbose (boolean): Specifies whether to print what experiment is currently running.
            load_bar (boolean): Specifies whether to show a loading bar while the experiments are running.
        '''

        self.problems, self.methods, self.problem_to_methods, self.metrics = to_dict(problems), to_dict(methods), problem_to_methods, metrics
        self.key, self.use_precomputed, self.timesteps, self.verbose, self.load_bar = key, use_precomputed, timesteps, verbose, load_bar

        self.n_problems, self.n_methods = {}, {}

        if(use_precomputed):

            if(timesteps > precomputed.get_timesteps()):
                print("WARNING: when using precomputed results, the maximum number of timesteps is fixed. " + \
                    "Will use %d instead of the specified %d" % (precomputed.get_timesteps(), timesteps))
                self.timesteps = precomputed.get_timesteps()

            # ensure problems and methods don't have specified hyperparameters
            if(problems is dict):
                print("WARNING: when using precomputed results, " + \
                      "any specified problem hyperparameters will be disregarded and default ones will be used instead.")
            
            if(methods is dict):
                print("WARNING: when using precomputed results, " + \
                      "any specified method hyperparameters will be disregarded and default ones will be used instead.")

            # map of the form [metric][problem][method] -> loss series + time + memory
            self.prob_method_to_result = precomputed.load_prob_method_to_result(problem_ids = list(self.problems.keys()), \
                                            method_ids = list(self.methods.keys()), problem_to_methods = problem_to_methods, metrics = metrics)

        else:
            self.new_experiment = NewExperiment()
            self.new_experiment.initialize(self.problems, self.methods, problem_to_methods, metrics, key, timesteps, verbose, load_bar)
            # map of the form [metric][problem][method] -> loss series + time + memory
            self.prob_method_to_result = self.new_experiment.run_all_experiments()

    def add_method(self, method_id, method_params = None, name = None):
        '''
        Description: Add a new method to the experiment instance.
        
        Args:
            method_id (string): ID of new method.
            method_params (dict): Parameters to use for initialization of new method.
        '''
        assert method_id is not None, "ERROR: No Method ID given."

        ### IS THIS USEFUL OR BAD ? ###
        if name is None and 'optimizer' in method_params:
            name = method_params['optimizer'].__name__

        new_id = ''
        if(method_id in self.methods):
            if(method_id not in self.n_methods):
                self.n_methods[method_id] = 0
            if(name is not None):
                new_id = method_id + '-' + name
            else:
                self.n_methods[method_id] += 1
                new_id = method_id + '-' + str(self.n_methods[method_id])
            self.methods[method_id].append((new_id, method_params))
        else:
            new_id = method_id
            if(name is not None):
                new_id += '-' + name
            self.methods[method_id] = [(new_id, method_params)]

        if(self.use_precomputed):
            print("WARNING: In precomputed mode, experiments for a new method will run for the predetermined key.")
            key = precomputed.get_key()
        else:
            key = self.key

        ''' Evaluate performance of new method on all problems '''
        for metric in self.metrics:
            for problem_id in self.problems.keys():
                for (new_problem_id, problem_params) in self.problems[problem_id]:

                    ''' If method is compatible with problem, run experiment and store results. '''
                    try:
                        loss, time, memory = run_experiment((problem_id, problem_params), (method_id, method_params), metric, \
                                        key = key, timesteps = self.timesteps, verbose = self.verbose, load_bar = self.load_bar)
                    except Exception as e:
                        print("ERROR: Could not run %s on %s. Please make sure method and problem are compatible." % (method_id, problem_id))
                        print(e)
                        loss, time, memory = 0, 0.0, 0.0

                    self.prob_method_to_result[(metric, new_problem_id, new_id)] = loss
                    self.prob_method_to_result[('time', new_problem_id, new_id)] = time
                    self.prob_method_to_result[('memory', new_problem_id, new_id)] = memory

    def add_problem(self, problem_id, problem_params = None, name = None):
        '''
        Description: Add a new problem to the experiment instance.
        
        Args:
            problem_id (string): ID of new method.
            problem_params (dict): Parameters to use for initialization of new method.
        '''
        assert problem_id is not None, "ERROR: No Problem ID given."

        new_id = ''
        if(problem_id in self.problems):
            if(problem_id not in self.n_problems):
                self.n_problems[problem_id] = 0
            if(name is not None):
                new_id = problem_id[:-2] + name
            else:
                self.n_problems[problem_id] += 1
                new_id = problem_id + '-' + str(self.n_problems[problem_id])
            self.problems[problem_id].append((new_id, problem_params))
        else:
            new_id = problem_id[:-2]
            if(name is not None):
                new_id += name
            self.problems[problem_id] = [(new_id, problem_params)]

        if(self.use_precomputed):
            print("WARNING: In precomputed mode, experiments for a new method will run for the predetermined key.")
            key = precomputed.get_key()
        else:
            key = self.key

        ''' Evaluate performance of new method on all problems '''
        for metric in self.metrics:
            for method_id in self.methods.keys():
                for (new_method_id, method_params) in self.methods[method_id]:

                    ''' If method is compatible with problem, run experiment and store results. '''
                    try:
                        loss, time, memory = run_experiment((problem_id, problem_params), (method_id, method_params), metric, \
                                        key = key, timesteps = self.timesteps, verbose = self.verbose, load_bar = self.load_bar)
                    except Exception as e:
                        print("ERROR: Could not run %s on %s. Please make sure method and problem are compatible." % (method_id, problem_id))
                        print(e)
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

        problem_ids = get_ids(self.problems)
        method_ids = get_ids(self.methods)

        table_dict['Problems'] = problem_ids

        field_names = ['Method\Problems']
        for problem_id in problem_ids:
            if(truncate_ids and len(problem_id) > 9):
                field_names.append(problem_id[:4] + '..' + problem_id[-3:])
            else:
                field_names.append(problem_id)

        table.field_names = field_names

        for method_id in method_ids:
            method_scores = [method_id]
            # get scores for each problem
            for problem_id in problem_ids:
                score = np.mean((self.prob_method_to_result\
                    [(metric, problem_id, method_id)])[start_time:self.timesteps])
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

    def _plot(self, ax, problem, problem_result_plus_method, n_problems, metric, \
                avg_regret, start_time, cutoffs, yscale, show_legend = True):

        for (loss, method) in problem_result_plus_method:
            if(avg_regret):
                ax.plot(self.avg_regret(loss[start_time:self.timesteps]), label=str(method))
            else:
                ax.plot(loss, label=str(method))
        if(show_legend):
            ax.legend(loc="upper right", fontsize=5 + 5//n_problems)
        ax.set_title("Problem:" + str(problem))
        #ax.set_xlabel("timesteps")
        ax.set_ylabel(metric)

        if(cutoffs is not None and problem in cutoffs.keys()):
            ax.set_ylim([0, cutoffs[problem]])

        if(yscale is not None):
            ax.set_yscale(yscale)

        return ax

    def graph(self, problem_ids = None, metric = 'mse', avg_regret = True, start_time = 0, \
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

        # get problem and method ids
        if(problem_ids is None):
            problem_ids = get_ids(self.problems)
        method_ids = get_ids(self.methods)

        # get number of problems
        n_problems = len(problem_ids)

        all_problem_info = []

        for problem_id in problem_ids:
            problem_result_plus_method = []
            method_list = []
            for method_id in method_ids:
                method_list.append(method_id)
                problem_result_plus_method.append((self.prob_method_to_result[(metric, problem_id, method_id)], method_id))
            all_problem_info.append((problem_id, problem_result_plus_method, method_list))

        nrows = max(int(np.sqrt(n_problems)), 1)
        ncols = n_problems // nrows + n_problems % nrows

        fig, ax = plt.subplots(figsize = (ncols * size, nrows * size), nrows=nrows, ncols=ncols)
        fig.canvas.set_window_title('TigerBench')

        if n_problems == 1:
            (problem, problem_result_plus_method, method_list) = all_problem_info[0]
            ax = self._plot(ax, problem, problem_result_plus_method, n_problems, \
                metric, avg_regret, start_time, cutoffs, yscale)
        elif nrows == 1:
            for j in range(ncols):
                (problem, problem_result_plus_method, method_list) = all_problem_info[j]
                ax[j] = self._plot(ax[j], problem, problem_result_plus_method, n_problems, \
                                          metric, avg_regret, start_time, cutoffs, yscale)
        else:
            cur_pb = 0
            for i in range(nrows):
                for j in range(ncols):

                    if(cur_pb == n_problems):
                        legend = []
                        for method_id in method_ids:
                            legend.append((0, method_id))
                        ax[i, j] = self._plot(ax[i, j], 'LEGEND', legend,\
                                n_problems, metric, False, cutoffs, None, show_legend = True)
                        continue

                    if(cur_pb > n_problems):
                        ax[i, j].plot(0, 'x', 'red', label="NO MORE \n MODELS")
                        ax[i, j].legend(loc="center", fontsize=8 + 10//n_problems)
                        continue

                    (problem, problem_result_plus_method, method_list) = all_problem_info[cur_pb]
                    cur_pb += 1
                    ax[i, j] = self._plot(ax[i, j], problem, problem_result_plus_method,\
                                n_problems, metric, avg_regret, start_time, cutoffs, yscale, show_legend = False)

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
             a range of problems.

Methods:

    initialize(problems = None, methods = None, problem_to_methods = None, metrics = ['mse'],
               use_precomputed = True, timesteps = 100, verbose = True, load_bar = True):

        Description: Initializes the experiment instance. 

        Args:
            problems (dict/list): map of the form problem_id -> hyperparameters for problem or list of problem ids;
                                  in the latter case, default parameters will be used for initialization

            methods (dict/list): map of the form method_id -> hyperparameters for method or list of method ids;
                                in the latter case, default parameters will be used for initialization

            problem_to_methods (dict) : map of the form problem_id -> list of method_id.
                                       If None, then we assume that the user wants to
                                       test every method in method_to_params against every
                                       problem in problem_to_params

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
