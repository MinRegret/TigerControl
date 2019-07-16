
import ctsb
from ctsb.utils.experiment import Experiment
import jax.numpy as np
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage


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
    mem_usage = memory_usage((exp.run_all_experiments, [100]))

    end = time.time()
    if verbose:
        print(end - start)
        exp.plot_all_problem_results()
        exp.get_performance_metrics()
        print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
        print('Maximum memory usage: %s' % max(mem_usage))
    return

if __name__ == "__main__":
    test_experiment_initialize()

