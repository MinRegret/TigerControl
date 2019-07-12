
import ctsb
from ctsb.utils.experiment import Experiment
import jax.numpy as np
import matplotlib.pyplot as plt
import time

# run all experiment tests
def test_experiment(steps=1000, show=False):
    test_experiment_time_series(steps=steps, verbose=show)
    print("test_experiment passed")


def test_experiment_time_series(steps=1000, verbose=False):
    # test without problem_to_models
    exp = Experiment()
    MSE = lambda y_true, y_pred: (y_true - y_pred)**2
    problem_to_params = {'ARMA-v0' : {'p':3, 'q':3}, 
                         'SP500-v0' : {}} 
    model_to_params = {'LastValue': {},
                       'PredictZero': {},}
    exp.initialize(MSE, problem_to_params, model_to_params)
    start = time.time()
    exp.run_all_experiments(steps)
    if verbose:
        print("Runtime: {}".format(time.time() - start))
        exp.plot_all_problem_results()
        print("test_experiment_time_series passed")

if __name__ == "__main__":
    test_experiment(show=True)


