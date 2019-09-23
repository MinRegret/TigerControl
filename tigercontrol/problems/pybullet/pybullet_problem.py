# PyBullet Problems

from tigercontrol import error
from tigercontrol.problems import Problem
from tigercontrol.problems.pybullet.simulator import Simulator


class PyBulletProblem(Simulator):
    ''' Description: class for online control tests '''
  
    def get_simulator(self): # same as calling fork()
        return self.fork()
