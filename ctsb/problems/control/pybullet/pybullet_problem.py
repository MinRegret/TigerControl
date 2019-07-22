# Problem class
# Author: John Hallman

from ctsb import error
from ctsb.problems import Problem
import pybullet as p
from ctsb.problems.control.pybullet.simulator_wrapper import SimulatorWrapper

class PyBulletProblem(Problem):
    ''' Description: class for online control tests '''
    
    def __init__(self):
        self.initialized = False

    def initialize(self):
        self.initialized = True
        self.sim = None
        self.env = None

    def getState(self):
        return self.sim.getState()

    def loadState(self, id):
        p.restoreState(stateId=id)

    # disconnect from physics server, end simulation
    def disconnect(self):
        p.disconnect()
    
    def get_simulator(self):
        sim_copy = SimulatorWrapper(self.sim.getEnv())
        return sim_copy




