
import ctsb
from ctsb.problems.simulated.arma import ARMA
from ctsb.models.time_series.last_value import LastValue
from ctsb.models.time_series.predict_zero import PredictZero
from ctsb.utils.experiment import Experiment
import jax.numpy as np
import matplotlib.pyplot as plt

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
	prob_model_to_loss = exp.get_prob_model_to_loss()

	all_models = []
	all_loss_series = []
	for problem, model_to_loss in prob_model_to_loss.items():
		print(problem)
		for model, loss in model_to_loss.items():
			print(model)
			all_models.append(model)
			all_loss_series.append(loss)
	if show_plot:
		for loss in all_loss_series:
			plt.plot(loss)
		plt.title("Problem:" + str(problem) + " , Model:" + str(all_models))
		plt.show(block=False)
		plt.pause(10)
		plt.close()

	return

if __name__ == "__main__":
	test_arma_experiment(show_plot=True)