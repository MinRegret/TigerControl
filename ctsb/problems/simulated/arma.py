"""
Autoregressive moving-average
"""

import ctsb
import numpy as np
from ctsb.utils import seeding


class ARMA(ctsb.Problem):
	"""
	Simulates an autoregressive moving-average time-series.
	"""

	def __init__(self):
		self.initialized = False

	def initialize(self, p, q, c=None):
		"""
		Description:
			Randomly initialize the hidden dynamics of the system.
		Args:
			p (int/numpy.ndarray): Autoregressive dynamics. If type int then randomly
				initializes a Gaussian length-p vector with L1-norm bounded by 1.0. 
				If p is a 1-dimensional numpy.ndarray then uses it as dynamics vector.
			q (int/numpy.ndarray): Moving-average dynamics. If type int then randomly
				initializes a Gaussian length-q vector (no bound on norm). If p is a
				1-dimensional numpy.ndarray then uses it as dynamics vector.
			c (float): Default value follows a normal distribution. The ARMA dynamics 
				follows the equation x_t = c + AR-part + MA-part + noise, and thus tends 
				to be centered around mean c.
		Returns:
			The first value in the time-series
		"""
		self.initialized = True
		self.T = 0
		if type(p) == int:
			phi = np.random.normal(size=(p,))
			self.phi = 1.0 * phi / np.linalg.norm(phi, ord=1)
		else:
			assert len(p.shape) == 1
			self.phi = p
		if type(q) == int:
			self.psi = np.random.normal(size=(q,))
		else:
			assert len(q.shape) == 1
			self.psi = q
		self.p = self.phi.shape[0]
		self.q = self.psi.shape[0]
		self.c = np.random.normal() if c == None else c
		self.x = np.random.normal(size=(self.p,))
		self.noise = np.random.normal(size=(q,))
		return self.x[0]

	def step(self):
		"""
		Description:
			Moves the system dynamics one time-step forward.
		Args:
			None
		Returns:
			The next value in the ARMA time-series.
		"""
		assert self.initialized
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

	def hidden(self):
		"""
		Description:
			Return the hidden state of the system.
		Args:
			None
		Returns:
			(x, eps): The hidden state consisting of the last p x-values and the last q
			noise-values.
		"""
		assert self.initialized
		return (self.x, self.noise)

	def seed(self, seed=None):
		"""
		Description:
			Seeds the random number generator to produce deterministic, reproducible results. 
		Args:
			seed (int): Default value None. The number that determines the seed of the random
			number generator for the system.
		Returns:
			A list containing the resulting NumPy seed.
		"""
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def close(self):
		"""
		Not implemented
		"""
		pass

	def help(self):
		"""
		Description:
			Prints information about this class and its methods.
		Args:
			None
		Returns:
			None
		"""
		print(ARMA_help)


# string to print when calling help() method
ARMA_help = """

-------------------- *** --------------------

Id: ARMA-v0
Description: Simulates an autoregressive moving-average time-series.

Methods:

	initialize(p, q, c=None)
		Description:
			Randomly initialize the hidden dynamics of the system.
		Args:
			p (int/numpy.ndarray): Autoregressive dynamics. If type int then randomly
				initializes a Gaussian length-p vector with L1-norm bounded by 1.0. 
				If p is a 1-dimensional numpy.ndarray then uses it as dynamics vector.
			q (int/numpy.ndarray): Moving-average dynamics. If type int then randomly
				initializes a Gaussian length-q vector (no bound on norm). If p is a
				1-dimensional numpy.ndarray then uses it as dynamics vector.
			c (float): Default value follows a normal distribution. The ARMA dynamics 
				follows the equation x_t = c + AR-part + MA-part + noise, and thus tends 
				to be centered around mean c.
		Returns:
			The first value in the time-series

	step()
		Description:
			Moves the system dynamics one time-step forward.
		Args:
			None
		Returns:
			The next value in the ARMA time-series.

	hidden()
		Description:
			Return the hidden state of the system.
		Args:
			None
		Returns:
			(x, eps): The hidden state consisting of the last p x-values and the last q
			noise-values.

	seed(seed)
		Description:
			Seeds the random number generator to produce deterministic, reproducible results. 
		Args:
			seed (int): Default value None. The number that determines the seed of the random
			number generator for the system.
		Returns:
			A list containing the resulting NumPy seed.

	help()
		Description:
			Prints information about this class and its methods.
		Args:
			None
		Returns:
			None

-------------------- *** --------------------

"""


