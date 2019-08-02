# PyBullet Problems

from ctsb import error
from ctsb.problems import Problem
from ctsb.problems.pybullet.simulator import Simulator


class PyBulletProblem(Simulator):
    ''' Description: class for online control tests '''
  
    def get_simulator(self): # same as calling fork()
        return self.fork()
