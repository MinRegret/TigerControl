# Experiment class

import ctsb
from ctsb import error
from ctsb.experiments.core import run_experiment, get_ids
from ctsb.experiments.new_experiment import NewExperiment
import ctsb.experiments.precomputed as precomputed
from statistics import mean
import csv
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class Experiment(object):

    def __init__(self):
        self.initialized = False
        
    def initialize(self, problems = None, models = None, problem_to_models=None, metrics = ['mse'], use_precomputed = True, timesteps = 100):
        '''
            Description:
                Initializes the experiment instance. 
            Args:
                problems (dict): map of the form problem_id -> hyperparameters for problem
                models (dict): map of the form model_id -> hyperparameters for model
                problem_to_models (dict) : map of the form problem_id -> list of model_id. If None, then we assume that the
                user wants to test every model in model_to_params against every problem in problem_to_params
        '''

        self.problems, self.models, self.problem_to_models, self.metrics, self.use_precomputed, self.timesteps = problems, models, problem_to_models, metrics, use_precomputed, timesteps
        self.new_models = 0

        if(use_precomputed and timesteps != precomputed.get_timesteps()):
            print("WARNING: when using precomputed results, number of timesteps is fixed.")

        if(use_precomputed):

            # ensure problems and models don't have specified hyperparameters
            if(problems is dict):
                print("WARNING: when using precomputed results, any specified problem hyperparameters will be disregarded and default ones will be used instead.")
                self.problems = list(problems.keys())
            if(models is dict):
                print("WARNING: when using precomputed results, any specified model hyperparameters will be disregarded and default ones will be used instead.")
                self.models = list(models.keys())

            # map of the form [metric][problem][model] -> loss series + time + memory
            self.prob_model_to_result = precomputed.load_prob_model_to_result(problem_ids = problems, model_ids = models, problem_to_models = problem_to_models, metrics = metrics)

        else:

            new_experiment = NewExperiment()
            new_experiment.initialize(problems, models, problem_to_params, metrics)

            # map of the form [metric][problem][model] -> loss series + time + memory
            self.prob_model_to_result = {}

    def add_model(self, model_id = None, model_params = None):
        if(self.use_precomputed):
            print("Running new model on all problems...")
            ''' Evaluate performance of new model on all problems '''
            for metric in metrics:
                for problem_id, problem_params in self.problems:
                    loss, time, memory = run_experiment((problem_id, problem_params), (model_id, model_params), metric, key = precomputed.get_key(), timesteps = precomputed.get_timesteps()) # in core
                    self.prob_model_to_result[metric][model_id][problem_id] = loss
                    self.prob_model_to_result['time'][problem][model] = time
                    self.prob_model_to_result['memory'][problem][model] = memory

        else:
            self.models[model_id] = parameters

        self.new_models += 1

    def run(self):
        if(self.use_precomputed):
            print("We are in precomputed mode, so everything has already been run.")
        else:
            self.prob_model_to_result = new_experiment.compute_prob_model_to_loss(self.problems, self.models, self.timesteps)

    def scoreboard(self, save_as = None, metric = 'mse'):
        '''
        Description:
            Initializes the experiment instance. 
        Args:
            metric (string): 
        '''

        if(self.use_precomputed and metric == 'time' and self.new_models):
            print("WARNING: Time comparison between precomputed models and any added model may be irrelevant due to hardware differences.")

        print("Scoreboard for " + metric + ":")
        table = PrettyTable()
        table_dict = {}

        problem_ids = get_ids(self.problems)
        model_ids = get_ids(self.models)

        table_dict['Models'] = model_ids

        field_names = ['Problems\Models']
        for model_id in model_ids:
            field_names.append(model_id)
        table.field_names = field_names

        for problem_id in problem_ids:
            problem_scores = [problem_id]
            # get scores for each model
            for model_id in model_ids:
                problem_scores.append(np.mean(self.prob_model_to_result[(metric, problem_id, model_id)]))
            table.add_row(problem_scores)
            table_dict[problem_id] = problem_scores[1:]

        print(table)

        ''' Save to csv file '''
        if(save_as is not None):
            with open(save_as, 'w') as f:
                for key in table_dict.keys():
                    f.write("%s,%s\n" % (key, table_dict[key]))

    def graph(self, save_as = None, metric = 'mse', time = 5):
        '''
        Description:
            Initializes the experiment instance. 
        Args:
            metric (string): 
        '''

        # check metric exists
        assert metric in self.metrics

        # get number of problems
        if(type(self.problems) is dict):
            n_problems = len(self.problems.keys())
        else:
            n_problems = len(self.problems)

        all_problem_info = []

        problem_ids = get_ids(self.problems)
        model_ids = get_ids(self.models)

        for problem_id in problem_ids:
            model_ids = get_ids(self.models)
            problem_result_plus_model = []
            model_list = []
            for model_id in model_ids:
                model_list.append(model_id)
                problem_result_plus_model.append((self.prob_model_to_result[(metric, problem_id, model_id)], model_id))
            all_problem_info.append((problem_id, problem_result_plus_model, model_list))

        fig, ax = plt.subplots(nrows=n_problems, ncols=1)
        if n_problems == 1:
            (problem, problem_result_plus_model, model_list) = all_problem_info[0]
            for (loss, model) in problem_result_plus_model:
                ax.plot(loss, label=str(model))
                ax.legend(loc="upper left")
            ax.set_title("Problem:" + str(problem))
            ax.set_xlabel("timesteps")
            ax.set_ylabel("loss")
        else:
            for i in range(n_problems):
                (problem, problem_result_plus_model, model_list) = all_problem_info[i]
                for (loss, model) in problem_result_plus_model:
                    ax[i].plot(loss, label=str(model))
                ax[i].set_title("Problem:" + str(problem), size=10)
                ax[i].legend(loc="upper left")
                ax[i].set_xlabel("timesteps")
                ax[i].set_ylabel("loss")

        fig.tight_layout()

        if time:
            plt.show(block=False)
            plt.pause(time)
            plt.close()
        else:
            plt.show()

        if(save_as is not None):
            plt.savefig(save_as)

    def help(self):
        # prints information about this class and its methods
        raise NotImplementedError


