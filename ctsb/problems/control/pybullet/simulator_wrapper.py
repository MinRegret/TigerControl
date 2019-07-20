# from ctsb.problems.control.pybullet.pybullet_problem import PyBulletProblem
import pybullet as p

class SimulatorWrapper(object):

    def __init__(self):
        self.initialized = False
        self._env = None

    def initialize(self, env):
        self.initialized = True
        self._env = env

    # state saving and loading methods
    def saveFile(self, filename):
        p.saveBullet(filename)

    def loadFile(self, name):
        p.restoreState(fileName = name)
        self.updateState()

    def getState(self):
        stateID = p.saveState()
        return stateID

    def loadState(self, ID):
        p.restoreState(stateId = ID)

    # gym environment methods
    def reset(self):
        return self._env.reset()

    def render(self, mode='human', close=False):
        self._env.render(mode=mode, close=close)

    def step(self, action):
        assert self.initialized
        return self._env.step(action)

    def get_observation_space(self):
        return self._env.observation_space.shape

    def get_action_space(self):
        return self._env.action_space.shape

    # return clone of simulator
    def fork(self):
        return SimulatorWrapper(self._env)
