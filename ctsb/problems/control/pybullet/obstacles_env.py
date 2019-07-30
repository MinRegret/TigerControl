import gym
import pybullet as pybullet
import numpy as np
import pybullet as pybullet
import gym.spaces as spaces
from ctsb.problems.control.pybullet.obstacle_utils import *

class ObstaclesEnv(gym.Env):
    def __init__(self, renders=True):
    # start the bullet physics server
        self._renders = renders
        if (renders):
          pybullet.connect(pybullet.GUI)
        else:
          pybullet.connect(pybullet.DIRECT)

        self.observation_space = spaces.Dict({"x_position": spaces.Box(low=-5.0, high=5.0, shape=(1,)), 
                                              "y_position": spaces.Box(low=0.0, high=10.0, shape=(1,)),
                                              "theta": spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(1,))})

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        # self.reset()

    def step_fgm(self, angle):
        # State: [x,y,theta]
        # x: horizontal position
        # y: vertical position
        # theta: angle from vertical (positive is anti-clockwise)
        
        # Dynamics:
        # xdot = -(r/2)*(ul + ur)*sin(theta)
        # ydot = (r/2)*(ul + ur)*cos(theta)
        # thetadot = (r/L)*(ur - ul)

        # Robot parameters
        r = 0.1; # Radius of robot wheel
        L = 0.5; # Length between wheels (i.e., width of base)
        
        dt = 0.05
        v0 = 2.5 # forward speed

        self.state[0] = self.state[0] + dt*(-(r/2)*(2*v0/r)*np.sin(self.state[2])) # x position
        self.state[1] = self.state[1] + dt*((r/2)*(2*v0/r)*np.cos(self.state[2])) # y position
        self.state[2] = angle

        # Update position of pybullet object
        quat = pybullet.getQuaternionFromEuler([0.0, 0.0, self.state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2
        pybullet.resetBasePositionAndOrientation(self.husky, [self.state[0], self.state[1], 0.0], quat)
        pybullet.resetBasePositionAndOrientation(self.sphere, [self.state[0], self.state[1], self.robotHeight], [0,0,0,1]) 

        # closestPoints = pybullet.getClosestPoints(self.sphere, self.obsUid, 0.0)
        # See if the robot is in collision. If so, cost = 1.0. 
        cost = 0.0
        closestPoints = pybullet.getClosestPoints(self.sphere, self.obsUid, 0.0)
        if closestPoints: # Check if closestPoints is non-empty 
            cost = 1.0
            # break # break out of simulation for this environment
        
        done =  self.state[0] < -5 \
                or self.state[0] > 5 \
                or self.state[1] < 0 \
                or self.state[1] > 10 \
                or self.state[2] < -np.pi/2 \
                or self.state[2] > np.pi/2 \
        
        done = bool(done)
        return self.state, cost, done, {}

    def step(self, u_diff):
        # State: [x,y,theta]
        # x: horizontal position
        # y: vertical position
        # theta: angle from vertical (positive is anti-clockwise)
        
        # Dynamics:
        # xdot = -(r/2)*(ul + ur)*sin(theta)
        # ydot = (r/2)*(ul + ur)*cos(theta)
        # thetadot = (r/L)*(ur - ul)

        # Robot parameters
        r = 0.1; # Radius of robot wheel
        L = 0.5; # Length between wheels (i.e., width of base)
        
        dt = 0.05
        v0 = 2.5 # forward speed
        
        # Saturate udiff
        u_diff_max = 0.5*(v0/r) 
        u_diff_min = -u_diff_max
        u_diff = np.maximum(u_diff_min, u_diff)
        u_diff = np.minimum(u_diff_max, u_diff)
        
        ul = v0/r - u_diff;
        ur = v0/r + u_diff;
        
        self.state[0] = self.state[0] + dt*(-(r/2)*(ul + ur)*np.sin(self.state[2])) # x position
        self.state[1] = self.state[1] + dt*((r/2)*(ul + ur)*np.cos(self.state[2])) # y position
        self.state[2] = self.state[2] + dt*((r/L)*(ur - ul))

        # Update position of pybullet object
        quat = pybullet.getQuaternionFromEuler([0.0, 0.0, self.state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2
        pybullet.resetBasePositionAndOrientation(self.husky, [self.state[0], self.state[1], 0.0], quat)
        pybullet.resetBasePositionAndOrientation(self.sphere, [self.state[0], self.state[1], self.robotHeight], [0,0,0,1]) 

        # closestPoints = pybullet.getClosestPoints(self.sphere, self.obsUid, 0.0)
        # See if the robot is in collision. If so, cost = 1.0. 
        cost = None
        closestPoints = pybullet.getClosestPoints(self.sphere, self.obsUid, 0.0)
        if closestPoints: # Check if closestPoints is non-empty 
            cost = 1.0
            # break # break out of simulation for this environment
        
        done =  self.state[0] < -5 \
                or self.state[0] > 5 \
                or self.state[1] < 0 \
                or self.state[1] > 10 \
                or self.state[2] < -np.pi/3 \
                or self.state[2] > np.pi/3 \
        
        done = bool(done)
        return self.state, cost, done, {}

    def reset(self):
        # print("-----------reset simulation---------------")
        pybullet.resetSimulation()
        # Get some robot parameters
        params = get_parameters()
        robotRadius = params['robotRadius']
        numRays = params['numRays']
        thetas_nominal = params['thetas_nominal']
        self.robotHeight = params['robotHeight']

        # Ground plane
        pybullet.loadURDF("./../pybullet/URDFs/plane.urdf")

        # Load robot from URDF
        self.husky = pybullet.loadURDF("./../pybullet/URDFs/husky.urdf", globalScaling=0.5)

        # Sphere
        colSphereId = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=robotRadius)
        mass = 0
        if (self._renders):
            visualShapeId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=robotRadius,rgbaColor=[0,0,0,0]) # This just makes sure that the sphere is not visible (we only use the sphere for collision checking)
        else:
            visualShapeId = -1

        self.sphere = pybullet.createMultiBody(mass,colSphereId,visualShapeId)

        # generate obstacles
        heightObs = 20*self.robotHeight
        self.obsUid = generate_obstacles(pybullet, heightObs, robotRadius)

        # Initialize position of robot
        self.state = [0.0, 1.0, 0.0] # [x, y, theta]
        quat = pybullet.getQuaternionFromEuler([0.0, 0.0, self.state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2

        pybullet.resetBasePositionAndOrientation(self.husky, [self.state[0], self.state[1], 0.0], quat)
        pybullet.resetBasePositionAndOrientation(self.sphere, [self.state[0], self.state[1], self.robotHeight], [0,0,0,1])

        return (np.array(self.state), params, self.husky, self.sphere, self.obsUid)

    def render(self, mode='human', close=False):
        return