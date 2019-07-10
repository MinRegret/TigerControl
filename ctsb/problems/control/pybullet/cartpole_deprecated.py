# CartPole problem class
# Author: Sandun Bambarandage

import os
import pybullet as p
import pybullet_data
from ctsb.problems.control.pybullet.pybullet_problem import PyBulletProblem

class CartPole(PyBulletProblem):

    def __init__(self):
        self._initialized = False

    def initialize(self, graphical=True, timeStep = 1. / 240):
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
    # run time.sleep(<timeStep>) after if using GUI
    def step(self, force):
        p.setJointMotorControl2(self._cartpole, 0, p.TORQUE_CONTROL, force=force)
        p.stepSimulation()
        self.updateState()

    # return current joint state
    def getState(self):
        stateList = []
        for joint in range(self._numJoints):
            stateList.append({"pos": self._state[joint][0], 
                              "vel": self._state[joint][1],
                              "force": self._state[joint][3],
                              "rForces": self._state[joint][2]})
        return stateList 
        
    # helper function, save state of all joints
    def updateState(self):
        joints = list(range(self._numJoints))
        self._state = p.getJointStates(self._cartpole, joints)


if __name__ == '__main__':
    # debug
    c = CartPole()   
    c.initialize()

    num = c.saveToMemory()
    import time
    for i in range(240 * 1):
        if i % 2: c.step(10000)
        else: c.step(-10000)
        time.sleep(1. / 240)
    
    print("getState")
    for s in c.getState():
        print(s)

    print("loadFromMemory then getState")
    c.loadFromMemory(num)
    for s in c.getState():
        print(s) 
    p.disconnect()


