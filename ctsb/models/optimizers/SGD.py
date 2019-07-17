'''SGD optimizer'''
from ctsb.models.optimizers.core import Optimizer
import jax
import jax.numpy as np
import jax.experimental.stax as stax

class SGD(Optimizer):
	def __init__(self, pred, loss, learning_rate):
		self.lr = learning_rate
		loss_fn = lambda model_params, a, b : loss(pred(model_params, a), b)
		self.grad_fn = jax.jit(jax.grad(loss_fn))

	def update(self, x, y_true, model_params):
		grad = self.grad_fn(model_params, x, y_true)
		return [w - self.lr * dw for (w,dw) in zip(model_params, grad)]


