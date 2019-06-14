"""
Produces an ARMA sequence based on given hyperparameters p,q.
"""

import ctsb
import numpy as np

# class for online control tests
class LDS(ctsb.Problem):

	# n: input dimension. m: observation dimension. d: LDS hidden dimension
	def __init__(self, n, m, d):
		self.T = 0
		self.n, self.m, self.d = n, m, d

		# shrinks matrix M such that largest eigenvalue has magnitude k
		normalize = lambda M, k: k * M / np.linalg.norm(M, ord=2)

		# initialize matrix dynamics
		self.A = np.random.normal(size=(d, d))
		self.B = np.random.normal(size=(d, n))
		self.C = np.random.normal(size=(m, d))
		self.D = np.random.normal(size=(m, n))
		self.h = np.random.normal(size=(d,))

		# adjust dynamics matrix A
		self.A = normalize(self.A, 1.0)
		self.B = normalize(self.B, 1.0)
		self.C = normalize(self.C, 1.0)
		self.D = normalize(self.D, 1.0)

	def seed(self, seed=None):
		#self.np_random, seed = seeding.np_random()
		return 1 #[seed]

	def step(self, u):
		assert u.shape == (self.n,)
		self.T += 1
		self.h = np.dot(self.A, self.h) + np.dot(self.B, u) + np.random.normal(self.d,)
		y = np.dot(self.C, self.h) + np.dot(self.D, u) + np.random.normal(self.m,)
		return y

	def reset(self):
		self.T = 0
		self.h = np.random.normal(size=(d,))

	def hidden(self):
		return self.h

	def close(self):
		pass