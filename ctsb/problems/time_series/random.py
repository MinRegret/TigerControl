"""
Produces randomly generated scalar values at every timestep, taken from a normal distribution.
"""

import ctsb
import numpy as np

class Random(ctsb.Problem):
	"""
	Description:
	    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
	Observation: 
	    Type: Scalar
	    Random value
	Actions:
	    Type: None
	"""

	def __init__(self):
		self.T = 0
		self.seed()

	def seed(self, seed=None):
		#self.np_random, seed = seeding.np_random()
		return 1 #[seed]

	def step(self):
		self.T += 1
		return np.random.normal(size=(1,))

	def reset(self):
		self.T = 0

	def hidden(self):
		pass

	def close(self):
		pass