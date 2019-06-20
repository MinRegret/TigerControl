# Problem class
# Author: John Hallman

import numpy
from ctsb import error

# class for online control tests
class Model(object):
	action_space = None
	observation_space = None
	metadata = {'render.modes': []}
	spec = None

	def initialize(self, **kwargs):
		# resets problem to time 0
		raise NotImplementedError

	def step(self, x=None, y=None):
		# run one timestep of the problem's dynamics
		raise NotImplementedError

	def predict(self, x=None):
		# run one timestep of the problem's dynamics
		raise NotImplementedError

	def update(self, rule=None):
		# run one timestep of the problem's dynamics
		raise NotImplementedError

	def help(self):
		# prints information about this class and its methods
		raise NotImplementedError

	@property
	def unwrapped(self):
		"""Completely unwrap this problem.
		Returns:
		    ctsb.Problem: The base non-wrapped ctsb.Problem instance
		"""
		return self

	def __str__(self):
		if self.spec is None:
			return '<{} instance> call object help() method for info'.format(type(self).__name__)
		else:
			return '<{}<{}>> call object help() method for info'.format(type(self).__name__, self.spec.id)

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()
		# propagate exception
		return False


