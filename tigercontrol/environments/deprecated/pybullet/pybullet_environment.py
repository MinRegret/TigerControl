# PyBullet Environments

from tigercontrol import error
from tigercontrol.environments import Environment
from tigercontrol.environments.deprecated.pybullet.simulator import Simulator


class PyBulletEnvironment(Simulator):
    ''' Description: class for online control tests '''
  
    def get_simulator(self): # same as calling fork()
        return self.fork()
