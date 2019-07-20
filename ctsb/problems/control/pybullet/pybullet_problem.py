# Problem class
# Author: John Hallman

from ctsb import error
from ctsb.problems import Problem
import pybullet as p
from ctsb.problems.control.pybullet.simulator_wrapper import SimulatorWrapper

# inherits the following methods from SimulatorWrapper:
# initialize, saveFile, loadFile, getState, loadState, reset, render, step
# get_observation_space, get_action_space, fork
class PyBulletProblem(SimulatorWrapper):

    # same as calling fork()
    def get_simulator(self):
        return self.fork()
