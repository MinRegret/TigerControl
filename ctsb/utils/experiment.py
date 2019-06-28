# Experiment class
# Author: John Hallman

import ctsb
from ctsb import error
import jax.numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# class for implementing algorithms with enforced modularity
class Experiment(object):

    def __init__(self, problem_id=None, problem_params=None, model_list=None, loss_fn=None, problem_to_param_models=None):
        self.initialized = False
        
    def initialize(self, loss_fn, problem_id=None, problem_params=None, model_id_list=None, problem_to_param_models=None):
        '''
        There are two ways to specify initialize(). The first is to use the params (problem_id, problem_params, model_id_list, loss_fn)
        and the second is to use the params (problem_to_param_models, loss_fn). The first way is useful if you only want to specify
        a single problem, and the second way is useful if you want to specify multiple problems.
        Args:
            problem_id(string): name of problem in registry
            problem_params(dict): hyperparameters for problem
            model_id_list(list of strings): list of model_id to run on problem
            loss_fn(function): function mapping (predict_value, true_value) -> loss
            problem_to_param_models(dict): map of the form problem_id -> (hyperparameters for problem, model list)
        '''
        self.intialized = True
        self.T = 0
        self.loss = loss_fn
        self.pom_ls = [] # (problem, initial observation, model) list
        self.prob_model_to_loss = {} # map of the form [problem][model] -> loss series

        if problem_id != None:
            problem = ctsb.problem(problem_id)
            x_0 = problem.initialize(**problem_params)
            model_list = []
            for model_id in model_id_list:
                model = ctsb.model(model_id)
                model.initialize()
                model_list.append(model)
            self.pom_ls = [(problem, x_0, model_list)]
            
        else:
            for problem_id, param_models in problem_to_param_models.items():
                problem = ctsb.problem(problem_id)
                x_0 = problem.initialize(**param_models[0])
                model_list = []
                for model_id in param_models[1]:
                    model = ctsb.model(model_id)
                    model.initialize()
                    model_list.append(model)
                self.pom_ls.append((problem, x_0, model_list))

    def run_all_experiments(self, time_steps=100):
        self.T = time_steps
        for (problem, obs, models) in self.pom_ls:
            self.prob_model_to_loss[problem] = {}
            for model in models:
                self.prob_model_to_loss[problem][model] = []
                self.run_experiment(problem, obs, model)
        return

    def run_experiment(self, problem, obs, model):
        cur_x = obs
        print ("running experiment: " + str(model) + " on " + str(problem))
        for i in tqdm(range(0,self.T)):
            cur_y_pred = model.predict(cur_x)
            cur_y_true= problem.step()
            cur_loss = self.loss(cur_y_true, cur_y_pred)
            self.prob_model_to_loss[problem][model].append(cur_loss)
            model.update(cur_loss)
            cur_x = cur_y_true
        return

    '''
    def plot_single_problem_results(self, problem, model_list):
        for model in model_list:
            plt.plot(self.prob_model_to_loss[problem][model])
        plt.title("Problem:" + str(problem) + " , Models:" + str(model_list))
        plt.show(block=False)
        plt.pause(10)
        plt.close()
    '''

    def plot_all_problem_results(self):
        all_problem_info = []
        for problem, model_to_loss in self.prob_model_to_loss.items():
            # print(problem)
            problem_loss_plus_model = []
            model_list = []
            for model, loss in model_to_loss.items():
                # print(model)
                model_list.append(model)
                problem_loss_plus_model.append((loss, model))
            all_problem_info.append((problem, problem_loss_plus_model, model_list))

        fig, ax = plt.subplots(nrows=len(self.pom_ls), ncols=1)
        if len(self.pom_ls) == 1:
            (problem, problem_loss_plus_model, model_list) = all_problem_info[0]
            for (loss,model) in problem_loss_plus_model:
                ax.plot(loss, label=str(model))
                ax.legend(loc="upper left")
            ax.set_title("Problem:" + str(problem))
            ax.set_xlabel("timesteps")
            ax.set_ylabel("loss")
        else:
            for i in range(len(self.pom_ls)):
                (problem, problem_loss_plus_model, model_list) = all_problem_info[i]
                for (loss, model) in problem_loss_plus_model:
                    ax[i].plot(loss, label=str(model))
                ax[i].set_title("Problem:" + str(problem), size=10)
                ax[i].legend(loc="upper left")
                ax[i].set_xlabel("timesteps")
                ax[i].set_ylabel("loss")

        fig.tight_layout()
        plt.show(block=False)
        plt.pause(100)
        plt.close()        

    def get_prob_model_to_loss(self):
        return self.prob_model_to_loss

    def help(self):
        # prints information about this class and its methods
        raise NotImplementedError
    '''
    def __str__(self):
        if self.spec is None:
            return '<{} instance> call object help() method for info'.format(type(self).__name__)
        else:
            return '<{}<{}>> call object help() method for info'.format(type(self).__name__, self.spec.id)
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        # propagate exception
        return False
    '''


