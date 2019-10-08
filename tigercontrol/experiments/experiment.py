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
        
    def initialize(self, problems = None, models = None, problem_to_models = None, metrics = ['mse'], \
                   key = 0, use_precomputed = False, timesteps = 100, verbose = False, load_bar = False):
        '''
        Description: Initializes the experiment instance. 

        Args:
            problems (dict/list): map of the form problem_id -> hyperparameters for problem or list of problem ids;
                                  in the latter case, default parameters will be used for initialization
            models (dict/list): map of the form model_id -> hyperparameters for model or list of model ids;
                                in the latter case, default parameters will be used for initialization
            problem_to_models (dict) : map of the form problem_id -> list of model_id.
                                       If None, then we assume that the user wants to
                                       test every model in model_to_params against every
                                       problem in problem_to_params
            metrics (list): Specifies metrics we are interested in evaluating.
            use_precomputed (boolean): Specifies whether to use precomputed results.
            timesteps (int): Number of time steps to run experiment for
            verbose (boolean): Specifies whether to print what experiment is currently running.
            load_bar (boolean): Specifies whether to show a loading bar while the experiments are running.
        '''

        self.problems, self.models, self.problem_to_models, self.metrics = to_dict(problems), to_dict(models), problem_to_models, metrics
        self.key, self.use_precomputed, self.timesteps, self.verbose, self.load_bar = key, use_precomputed, timesteps, verbose, load_bar

        self.n_problems, self.n_models = {}, {}

        if(use_precomputed):

            if(timesteps > precomputed.get_timesteps()):
                print("WARNING: when using precomputed results, the maximum number of timesteps is fixed. " + \
                    "Will use %d instead of the specified %d" % (precomputed.get_timesteps(), timesteps))
                self.timesteps = precomputed.get_timesteps()

            # ensure problems and models don't have specified hyperparameters
            if(problems is dict):
                print("WARNING: when using precomputed results, " + \
                      "any specified problem hyperparameters will be disregarded and default ones will be used instead.")
            
            if(models is dict):
                print("WARNING: when using precomputed results, " + \
                      "any specified model hyperparameters will be disregarded and default ones will be used instead.")

            # map of the form [metric][problem][model] -> loss series + time + memory
            self.prob_model_to_result = precomputed.load_prob_model_to_result(problem_ids = list(self.problems.keys()), \
                                            model_ids = list(self.models.keys()), problem_to_models = problem_to_models, metrics = metrics)

        else:
            self.new_experiment = NewExperiment()
            self.new_experiment.initialize(self.problems, self.models, problem_to_models, metrics, key, timesteps, verbose, load_bar)
            # map of the form [metric][problem][model] -> loss series + time + memory
            self.prob_model_to_result = self.new_experiment.run_all_experiments()

    def add_model(self, model_id, model_params = None, name = None):
        '''
        Description: Add a new model to the experiment instance.
        
        Args:
            model_id (string): ID of new model.
            model_params (dict): Parameters to use for initialization of new model.
        '''
        assert model_id is not None, "ERROR: No Model ID given."

        ### IS THIS USEFUL OR BAD ? ###
        if name is None and 'optimizer' in model_params:
            name = model_params['optimizer'].__name__

        new_id = ''
        if(model_id in self.models):
            if(model_id not in self.n_models):
                self.n_models[model_id] = 0
            if(name is not None):
                new_id = model_id + '-' + name
            else:
                self.n_models[model_id] += 1
                new_id = model_id + '-' + str(self.n_models[model_id])
            self.models[model_id].append((new_id, model_params))
        else:
            new_id = model_id
            if(name is not None):
                new_id += '-' + name
            self.models[model_id] = [(new_id, model_params)]

        if(self.use_precomputed):
            print("WARNING: In precomputed mode, experiments for a new model will run for the predetermined key.")
            key = precomputed.get_key()
        else:
            key = self.key

        ''' Evaluate performance of new model on all problems '''
        for metric in self.metrics:
            for problem_id in self.problems.keys():
                for (new_problem_id, problem_params) in self.problems[problem_id]:

                    ''' If model is compatible with problem, run experiment and store results. '''
                    try:
                        loss, time, memory = run_experiment((problem_id, problem_params), (model_id, model_params), metric, \
                                        key = key, timesteps = self.timesteps, verbose = self.verbose, load_bar = self.load_bar)
                    except Exception as e:
                        print("ERROR: Could not run %s on %s. Please make sure model and problem are compatible." % (model_id, problem_id))
                        print(e)
                        loss, time, memory = 0, 0.0, 0.0

                    self.prob_model_to_result[(metric, new_problem_id, new_id)] = loss
                    self.prob_model_to_result[('time', new_problem_id, new_id)] = time
                    self.prob_model_to_result[('memory', new_problem_id, new_id)] = memory

    def add_problem(self, problem_id, problem_params = None, name = None):
        '''
        Description: Add a new problem to the experiment instance.
        
        Args:
            problem_id (string): ID of new model.
            problem_params (dict): Parameters to use for initialization of new model.
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
            print("WARNING: In precomputed mode, experiments for a new model will run for the predetermined key.")
            key = precomputed.get_key()
        else:
            key = self.key

        ''' Evaluate performance of new model on all problems '''
        for metric in self.metrics:
            for model_id in self.models.keys():
                for (new_model_id, model_params) in self.models[model_id]:

                    ''' If model is compatible with problem, run experiment and store results. '''
                    try:
                        loss, time, memory = run_experiment((problem_id, problem_params), (model_id, model_params), metric, \
                                        key = key, timesteps = self.timesteps, verbose = self.verbose, load_bar = self.load_bar)
                    except Exception as e:
                        print("ERROR: Could not run %s on %s. Please make sure model and problem are compatible." % (model_id, problem_id))
                        print(e)
                        loss, time, memory = 0.0, 0.0, 0.0

                    self.prob_model_to_result[(metric, new_id, new_model_id)] = loss
                    self.prob_model_to_result[('time', new_id, new_model_id)] = time
                    self.prob_model_to_result[('memory', new_id, new_model_id)] = memory

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

        if(self.use_precomputed and metric == 'time' and len(self.n_models.keys()) > 0):
            print("WARNING: Time comparison between precomputed models and" + \
                  "any added model may be irrelevant due to hardware differences.")

        if(verbose and metric in self.metrics):
            print("Average " + metric + ":")
        else:
            print(metric + ":")
            
        table = PrettyTable()
        table_dict = {}

        problem_ids = get_ids(self.problems)
        model_ids = get_ids(self.models)

        table_dict['Problems'] = problem_ids

        field_names = ['Model\Problems']
        for problem_id in problem_ids:
            if(truncate_ids and len(problem_id) > 9):
                field_names.append(problem_id[:4] + '..' + problem_id[-3:])
            else:
                field_names.append(problem_id)

        table.field_names = field_names

        for model_id in model_ids:
            model_scores = [model_id]
            # get scores for each problem
            for problem_id in problem_ids:
                score = np.mean((self.prob_model_to_result\
                    [(metric, problem_id, model_id)])[start_time:self.timesteps])
                score = round(float(score), n_digits)
                if(score == 0.0):
                    score = '—'
                model_scores.append(score)
            table.add_row(model_scores)
            table_dict[model_id] = model_scores[1:]

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

    def _plot(self, ax, problem, problem_result_plus_model, n_problems, metric, \
                avg_regret, start_time, cutoffs, yscale, show_legend = True):

        for (loss, model) in problem_result_plus_model:
            if(avg_regret):
                ax.plot(self.avg_regret(loss[start_time:self.timesteps]), label=str(model))
            else:
                ax.plot(loss, label=str(model))
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

        # get problem and model ids
        if(problem_ids is None):
            problem_ids = get_ids(self.problems)
        model_ids = get_ids(self.models)

        # get number of problems
        n_problems = len(problem_ids)

        all_problem_info = []

        for problem_id in problem_ids:
            problem_result_plus_model = []
            model_list = []
            for model_id in model_ids:
                model_list.append(model_id)
                problem_result_plus_model.append((self.prob_model_to_result[(metric, problem_id, model_id)], model_id))
            all_problem_info.append((problem_id, problem_result_plus_model, model_list))

        nrows = max(int(np.sqrt(n_problems)), 1)
        ncols = n_problems // nrows + n_problems % nrows

        fig, ax = plt.subplots(figsize = (ncols * size, nrows * size), nrows=nrows, ncols=ncols)
        fig.canvas.set_window_title('TigerBench')

        if n_problems == 1:
            (problem, problem_result_plus_model, model_list) = all_problem_info[0]
            ax = self._plot(ax, problem, problem_result_plus_model, n_problems, \
                metric, avg_regret, start_time, cutoffs, yscale)
        elif nrows == 1:
            for j in range(ncols):
                (problem, problem_result_plus_model, model_list) = all_problem_info[j]
                ax[j] = self._plot(ax[j], problem, problem_result_plus_model, n_problems, \
                                          metric, avg_regret, start_time, cutoffs, yscale)
        else:
            cur_pb = 0
            for i in range(nrows):
                for j in range(ncols):

                    if(cur_pb == n_problems):
                        legend = []
                        for model_id in model_ids:
                            legend.append((0, model_id))
                        ax[i, j] = self._plot(ax[i, j], 'LEGEND', legend,\
                                n_problems, metric, False, cutoffs, None, show_legend = True)
                        continue

                    if(cur_pb > n_problems):
                        ax[i, j].plot(0, 'x', 'red', label="NO MORE \n MODELS")
                        ax[i, j].legend(loc="center", fontsize=8 + 10//n_problems)
                        continue

                    (problem, problem_result_plus_model, model_list) = all_problem_info[cur_pb]
                    cur_pb += 1
                    ax[i, j] = self._plot(ax[i, j], problem, problem_result_plus_model,\
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
        return "<Experiment Model>"

# string to print when calling help() method
Experiment_help = """

-------------------- *** --------------------

Description: Streamlines the process of performing experiments and comparing results of models across
             a range of problems.

Methods:

    initialize(problems = None, models = None, problem_to_models = None, metrics = ['mse'],
               use_precomputed = True, timesteps = 100, verbose = True, load_bar = True):

        Description: Initializes the experiment instance. 

        Args:
            problems (dict/list): map of the form problem_id -> hyperparameters for problem or list of problem ids;
                                  in the latter case, default parameters will be used for initialization

            models (dict/list): map of the form model_id -> hyperparameters for model or list of model ids;
                                in the latter case, default parameters will be used for initialization

            problem_to_models (dict) : map of the form problem_id -> list of model_id.
                                       If None, then we assume that the user wants to
                                       test every model in model_to_params against every
                                       problem in problem_to_params

            metrics (list): Specifies metrics we are interested in evaluating.

            use_precomputed (boolean): Specifies whether to use precomputed results.

            timesteps (int): Number of time steps to run experiment for

            verbose (boolean): Specifies whether to print what experiment is currently running.

            load_bar (boolean): Specifies whether to show a loading bar while the experiments are running.


    add_model(model_id, model_params = None):

        Description: Add a new model to the experiment instance.
        
        Args:
            model_id (string): ID of new model.

            model_params: Parameters to use for initialization of new model.


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
