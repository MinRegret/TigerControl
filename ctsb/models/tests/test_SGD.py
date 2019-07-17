import ctsb
from ctsb.models.optimizers.losses import mse
from ctsb.utils.experiment import Experiment
def test_sgd_arma():
	problem = ctsb.problem('ARMA-v0')
	problem.initialize(p=2,q=0)
	exp = Experiment()
	# MSE = lambda y_true, y_pred: (y_true - y_pred)**2
	problem_to_params = {'ARMA-v0' : {'p':2, 'q':0}} 
	model_to_params = {'LSTM': {'n': 1, 'm': 1, 'l':3, 'h':10}}
	exp.initialize(mse, problem_to_params, model_to_params)
	exp.run_all_experiments(time_steps=3000)
	exp.plot_all_problem_results()

if __name__ == "__main__":
	test_sgd_arma()