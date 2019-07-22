# Problem class
# Author: John Hallman

from ctsb import error
from ctsb.problems import Problem
import pybullet as p
from ctsb.problems.control.pybullet.simulator_wrapper import SimulatorWrapper

class PyBulletProblem(SimulatorWrapper):
  ''' Description: class for online control tests '''
  
    def get_simulator(self): # same as calling fork()
        return self.fork()
