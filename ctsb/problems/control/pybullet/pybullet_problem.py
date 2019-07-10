# Problem class
# Author: John Hallman

from ctsb import error
from ctsb.problems import Problem
import pybullet as p

# class for online control tests
class PyBulletProblem(Problem):
    
    # save sim state to disk
    def saveToDisk(self, filename):
        p.saveBullet(filename)

    # load sim state from disk
    def loadFromDisk(self, name):
        p.restoreState(fileName = name)
        self.updateState()

    # save state to memory
    # keep track of state id
    def saveToMemory(self):
        stateID = p.saveState()
        return stateID

    # load state from memory
    def loadFromMemory(self, ID):
        p.restoreState(stateId = ID)
        # self.updateState()
        

    # disconnect from physics server, end simulation
    def disconnect(self):
        p.disconnect()



