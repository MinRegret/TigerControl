
from ctsb.problems.control.pybullet.pybullet_problem import PyBulletProblem

class SimulatorWrapper(PyBulletProblem):
	def __init__(self, problem, state_id):
		self.state_id = state_id
		self.problem = problem

	def set_state(self, state_id):
		self.state_id = state_id

	def step(self, action):
		return self.problem.step(action)