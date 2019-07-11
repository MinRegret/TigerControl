# Experiment class
# Author: Alex Yu

import ctsb
from ctsb import error
from ctsb.problems.control.control_problem import ControlProblem
from ctsb.models.control import ControlModel
import jax.numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import inspect
from jax import jit

# class for implementing algorithms with enforced modularity
class Experiment(object):

    def __init__(self, problem_id=None, problem_params=None, model_list=None, loss_fn=None, problem_to_param_models=None):
        self.initialized = False
        
    def initialize(self, loss_fn, problem_to_params, model_to_params, problem_to_models=None):
        '''
            Description:
                Initializes the experiment instance. 
            Args:
                loss_fn (function): function mapping (predict_value, true_value) -> loss
                problem_to_param (dict): map of the form problem_id -> hyperparameters for problem
                model_to_param (dict): map of the form model_id -> hyperparameters for model
                problem_to_models (dict) : map of the form problem_id -> list of model_id. If None, then we assume that the
                user wants to test every model in model_to_params against every problem in problem_to_params
        '''
        self.intialized = True
        self.T = 0
        self.loss = loss_fn
        self.pom_ls = [] # (problem, initial observation, model) list
        self.prob_model_to_loss = {} # map of the form [problem][model] -> loss series

        for problem_id, problem_params in problem_to_params.items():
            problem = ctsb.problem(problem_id)
            x_0 = problem.initialize(**problem_params)
            initialized_models = []
            model_list = problem_to_models[problem_id] if problem_to_models != None else list(model_to_params.keys())
            for model_id in model_list:
                model = ctsb.model(model_id)
                model.initialize(**model_to_params[model_id])
                initialized_models.append(model)
            self.pom_ls.append((problem, x_0, initialized_models))

    def run_all_experiments(self, time_steps=100):
        '''
        Descripton:
            Runs all experiments for specified number of timesteps.
        Args:
            time_steps (int): number of time steps 
        '''
        self.T = time_steps
        for (problem, obs, models) in self.pom_ls:
            self.prob_model_to_loss[problem] = {}
            for model in models:
                print("model:" + str(model))
                self.prob_model_to_loss[problem][model] = self.run_experiment(problem, obs, model)
        return

    def run_experiment(self, problem, obs, model):
        '''
        Descripton:
            Runs all experiments for specified number of timesteps.
        Args:
            problem (instance of ctsb.Problem): initialized problem
            obs (initial observation): initial observation
            model (instance of ctsb.Model): initialized model
        '''
        is_control_problem = (inspect.getmro(problem.__class__))[1] == ControlProblem
        is_control_model = (inspect.getmro(model.__class__))[1] == ControlModel
        assert ((is_control_problem and is_control_model) or (not is_control_problem and not is_control_model))
        # args = {'problem_step' : problem.step, 'obs' : obs, 'model_predict' : model.predict}
        cur_x = obs
        loss = []
        for i in tqdm(range(0,self.T)):
            cur_y_pred = model.predict(cur_x)
            cur_y_true = problem.step(cur_x) if is_control_problem else problem.step()
            cur_loss = self.loss(cur_y_true, cur_y_pred)
            loss.append(cur_loss)
            model.update(cur_y_true)
            cur_x = cur_y_true
        
        return loss

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
        plt.pause(5)
        plt.close()        

    def get_prob_model_to_loss(self):
        return self.prob_model_to_loss

    def help(self):
        # prints information about this class and its methods
        raise NotImplementedError


