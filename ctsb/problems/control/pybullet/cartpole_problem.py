# CartPole problem class
# Author: Sandun Bambarandage

import os
import pybullet as p
import pybullet_data

class CartPoleProblem():

    def __init__(self):
        self._initialized = False

    def initialize(self, graphical=True, timeStep = 0.01):
        if not self._initialized:
    
            # set render mode, time step
            self._graphical = graphical
            self._timeStep = timeStep

            # connect to physics client
            if self._graphical: self._physicsClient = p.connect(p.GUI)
            else: self._physicsClient = p.connect(p.DIRECT)

            # create cartpole
            self._cartpole = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "cartpole.urdf"), [0, 0, 0])
            self._numJoints = p.getNumJoints(self._cartpole)

            # remove damping on cartpole joints
            p.changeDynamics(self._cartpole, -1, linearDamping=0, angularDamping=0)
            p.changeDynamics(self._cartpole, 0, linearDamping=0, angularDamping=0)
            p.changeDynamics(self._cartpole, 1, linearDamping=0, angularDamping=0)

            # set world parameters
            p.setGravity(0, 0, -9.8)
            p.setTimeStep(self._timeStep)
            
            # disable default velocity motor to enable torque motor
            p.setJointMotorControl2(self._cartpole, 0, p.VELOCITY_CONTROL, force=0)
            p.setJointMotorControl2(self._cartpole, 1, p.VELOCITY_CONTROL, force=0)

            self.updateState()
            self._initialized = True

    # move one step in the simulation
    def step(self, force):
        p.setJointMotorControl2(self._cartpole, 0, p.TORQUE_CONTROL, force=force)
        p.stepSimulation()
        self.updateState()

    # helper function, save state of all joints
    def updateState(self):
        joints = list(range(self._numJoints))
        self._state = p.getJointStates(self._cartpole, joints)
    
    # save state to disk
    def saveState(self, path = None):
        pass

    # load state from disk
    def loadState(self, path):
        pass

    # return current state
    def getState(self):
        return self._state # format output at some point

# debug
c = CartPoleProblem()   
c.initialize()
for i in range(10000000):
    if i % 2: c.step(5)
    else: c.step(-5)



    

