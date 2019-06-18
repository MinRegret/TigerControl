"""
Linear dynamical system
"""

import ctsb
import numpy as np
from ctsb.utils import seeding


class LDS(ctsb.Problem):
	"""
	Simulates a linear dynamical system.
	"""

	def __init__(self):
		self.initialized = False

	def initialize(self, n, m, d, noise=1.0):
		"""
		Description:
			Randomly initialize the hidden dynamics of the system.
		Args:
			n (int): Input dimension.
			m (int): Observation/output dimension.
			d (int): Hidden state dimension.
			noise (float): Default value 1.0. The magnitude of the noise (Gaussian) added
				to both the hidden state and the observable output.
		Returns:
			The first value in the time-series
		"""
		self.initialized = True
		self.T = 0
		self.n, self.m, self.d, self.noise = n, m, d, noise

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

		y = np.dot(self.C, self.h) + np.dot(self.D, np.zeros(n)) + noise * np.random.normal(m,)
		return y


	def step(self, u):
		"""
		Description:
			Moves the system dynamics one time-step forward.
		Args:
			u (numpy.ndarray): control input, an n-dimensional real-valued vector.
		Returns:
			A new observation from the LDS.
		"""
		assert self.initialized
		assert u.shape == (self.n,)
		self.T += 1
		self.h = np.dot(self.A, self.h) + np.dot(self.B, u) + self.noise * np.random.normal(self.d,)
		y = np.dot(self.C, self.h) + np.dot(self.D, u) + self.noise * np.random.normal(self.m,)
		return y

	def hidden(self):
		"""
		Description:
			Return the hidden state of the system.
		Args:
			None
		Returns:
			h: The hidden state of the LDS.
		"""
		assert self.initialized
		return self.h

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
		print(LDS_help)



# string to print when calling help() method
LDS_help = """

-------------------- *** --------------------

Id: LDS-v0
Description: Simulates a linear dynamical system.

Methods:

	initialize(n, m, d, noise=1.0)
		Description:
			Randomly initialize the hidden dynamics of the system.
		Args:
			n (int): Input dimension.
			m (int): Observation/output dimension.
			d (int): Hidden state dimension.
			noise (float): Default value 1.0. The magnitude of the noise (Gaussian) added
				to both the hidden state and the observable output.
		Returns:
			The first value in the time-series

	step(u)
		Description:
			Moves the system dynamics one time-step forward.
		Args:
			u (numpy.ndarray): control input, an n-dimensional real-valued vector.
		Returns:
			A new observation from the LDS.

	hidden()
		Description:
			Return the hidden state of the system.
		Args:
			None
		Returns:
			h: The hidden state of the LDS.

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




