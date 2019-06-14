"""
Produces an ARMA sequence based on given hyperparameters p,q.
"""

import ctsb
import numpy as np

# class for online control tests
class ARMA(ctsb.Problem):

	# p: AR dimension. q: MA dimension.
	def __init__(self, p, q):
		self.c = np.random.normal()
		phi = np.random.normal(size=(p,))
		phi = 0.99 * phi / np.linalg.norm(phi, ord=1)
		self.phi = phi
		self.psi = np.random.normal(size=(q,))
		self.x = np.random.normal(size=(p,))
		self.noise = np.random.normal(size=(q))
		self.T = 0

	def seed(self, seed=None):
		#self.np_random, seed = seeding.np_random()
		return 1 #[seed]

	def step(self):
		self.T += 1
		x_ar = np.dot(self.x, self.phi)
		x_ma = np.dot(self.noise, self.psi)
		eps = np.random.normal()
		x_new = self.c + x_ar + x_ma + eps
		self.x[1:] = self.x[:-1]
		self.x[0] = x_new
		self.noise[1:] = self.noise[:-1]
		self.noise[0] = eps
		return x_new

	def reset(self):
		self.T = 0
		self.x = np.random.normal(size=(p,))
		self.noise = np.random.normal(size=(q))

	def hidden(self):
		return (self.x, self.noise)

	def close(self):
		pass