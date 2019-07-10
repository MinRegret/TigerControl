
from ctsb.problems.control.pybullet.pybullet_problem import PyBulletProblem

class simulator_wrapper(PyBulletProblem):
	def __init__(self, state):
		self.state = state

	def set_state(state):
		self.state = state