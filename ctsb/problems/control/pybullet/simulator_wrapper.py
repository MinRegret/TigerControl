
# from ctsb.problems.control.pybullet.pybullet_problem import PyBulletProblem
import pybullet as p

class SimulatorWrapper(object):
	def __init__(self, env):
		self.env = env

	def set_state_from(self, state_id):
		self.loadFromMemory(state_id)

	def saveFile(self, filename):
		p.saveBullet(filename)

	# load sim state from disk
	def loadFile(self, name):
		p.restoreState(fileName = name)
		self.updateState()

	# save state to memory
	# keep track of state id
	def getState(self):
		stateID = p.saveState()
		return stateID

	# load state from memory
	def loadState(self, ID):
		p.restoreState(stateId = ID)
		# self.updateState()

	def step(self, action):
		return self.env.step(action)