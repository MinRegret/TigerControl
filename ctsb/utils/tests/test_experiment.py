
import ctsb
from ctsb.problems.time_series import Crypto
from ctsb.problems.control.arma import ARMA
from ctsb.models.time_series.last_value import LastValue
from ctsb.models.time_series.predict_zero import PredictZero
from ctsb.utils.experiment import Experiment
import jax.numpy as np
import matplotlib.pyplot as plt

# run all experiment teps
def test_experiment(steps=100):
    test_single_problem_experiment_intialize(steps)
    test_multiple_problem_experiment_inialize(steps)
    print("test_experiment passed")

def test_single_problem_experiment_intialize(steps=100):
    exp = Experiment()
    MSE = lambda y_true, y_pred: (y_true - y_pred)**2
    exp.initialize(MSE, problem_id="ARMA-v0", problem_params={'p': 3, 'q':3}, model_id_list=['LastValue', 'PredictZero'])
    exp.run_all_experiments(steps)
    exp.plot_all_problem_results()
    return

def test_multiple_problem_experiment_inialize(steps=100):
    exp = Experiment()
    MSE = lambda y_true, y_pred: (y_true - y_pred)**2
    model_id_list = ['LastValue', 'PredictZero']
    exp.initialize(MSE, problem_to_param_models={'ARMA-v0' : ({'p':3, 'q':3}, model_id_list), 
                                                 'SP500-v0' : ({}, model_id_list)})
    exp.run_all_experiments(steps)
    exp.plot_all_problem_results()
    return
'''
def test_arma_and_crypto(steps=100, show_plot=False, verbose=False):
    T = steps
    p, q = 3, 3
    problem_arma = ARMA()
    x_0_arma = problem_arma.initialize(p,q)

    problem_crypto = Crypto()
    x_0_crypto = problem_crypto.initialize()

    last_value = LastValue()
    last_value.initialize()

    predict_zero = PredictZero()
    predict_zero.initialize()

    exp = Experiment()
    MSE = lambda y_true, y_pred: (y_true - y_pred)**2
    exp.initialize([(problem_arma, x_0_arma, [last_value, predict_zero]),
                    (problem_crypto, x_0_crypto, [last_value, predict_zero])], 
                    MSE, 
                    T)
    exp.run_all_experiments()
    if show_plot:
        exp.plot_all_problem_results()
    return

def test_arma_experiment(steps=100, show_plot=False, verbose=False):
    T = steps
    p, q = 3, 3
    problem = ARMA()
    x_0 = problem.initialize(p,q)

    last_value = LastValue()
    last_value.initialize()

    predict_zero = PredictZero()
    predict_zero.initialize()

    exp = Experiment()
    MSE = lambda y_true, y_pred: (y_true - y_pred)**2
    exp.initialize([(problem, x_0, [last_value, predict_zero])], MSE, T)
    exp.run_all_experiments()
    if show_plot:
        exp.plot_all_problem_results()
    return
'''

if __name__ == "__main__":
    test_experiment()
    # test_single_problem_experiment_intialize()
    # test_arma_and_crypto(show_plot=True)
    # test_arma_experiment(show_plot=True)