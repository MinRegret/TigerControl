import ctsb
from ctsb.models.optimizers.losses import mse
from ctsb.models.optimizers.SGD import SGD
from ctsb.experiments import Experiment

def test_sgd(time=1, show=False):
    problem = ctsb.problem('ARMA-v0')
    problem.initialize(p=2,q=0)
    exp = Experiment()
    # MSE = lambda y_true, y_pred: (y_true - y_pred)**2
    problem_to_params = {'ARMA-v0' : {'p':2, 'q':0}} 
    model_to_params = {'LSTM': {'n': 1, 'm': 1, 'l':3, 'h':10, 'optimizer': SGD, 'loss': mse}}
    exp.initialize(mse, problem_to_params, model_to_params)
    exp.run_all_experiments(time_steps=3000)
    if show:
        exp.plot_all_problem_results(time=time)
    print("test_sgd passed")

if __name__ == "__main__":
    test_sgd(time=3)