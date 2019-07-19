import ctsb
from ctsb.models.optimizers.losses import mse
from ctsb.models.optimizers.OGD import OGD
from ctsb.experiments import Experiment

def test_ogd(time=3, show=False):
    problem = ctsb.problem('ARMA-v0')
    problem.initialize(p=2,q=0)
    exp = Experiment()
    # MSE = lambda y_true, y_pred: (y_true - y_pred)**2
    problem_to_params = {'ARMA-v0' : {'p':2, 'q':0}} 
    model_to_params = {'ArmaOgd': {'p':2, 'optimizer': OGD, 'loss': mse}}
    exp.initialize(mse, problem_to_params, model_to_params)
    exp.run_all_experiments(time_steps=1000)
    if show:
        exp.plot_all_problem_results(time=time)

if __name__ == "__main__":
    test_ogd(time=3)