import gym
import pybullet as pybullet
import numpy as np
import pybullet as pybullet
import gym.spaces as spaces
# from tigercontrol.problems.pybullet.obstacle_utils import *
from tigercontrol.utils import get_tigercontrol_dir
from tigercontrol.problems.pybullet.pybullet_problem import PyBulletProblem
import os

class ObstaclesEnv(gym.Env):
    def __init__(self, renders=True, params=None):
    # start the bullet physics server
        self._renders = renders
        if (renders):
          pybullet.connect(pybullet.GUI)
        else:
          pybullet.connect(pybullet.DIRECT)

        self.observation_space = spaces.Dict({"x_position": spaces.Box(low=-5.0, high=5.0, shape=(1,)), 
                                              "y_position": spaces.Box(low=0.0, high=26.0, shape=(1,)),
                                              "theta": spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(1,))})

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        self.params = params if params != None else self.default_params()
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
                or self.state[2] < -np.pi/2 \
                or self.state[2] > np.pi/2 \
        
        done = bool(done)
        return self.state, cost, done, {}

    def reset(self):
        # print("-----------reset simulation---------------")
        pybullet.resetSimulation()
        # Get some robot parameters
        params = self.params
        robotRadius = params['robotRadius']
        numRays = params['numRays']
        thetas_nominal = params['thetas_nominal']
        self.robotHeight = params['robotHeight']

        # Ground plane
        tigercontrol_dir = get_tigercontrol_dir()
        pybullet.loadURDF(os.path.join(tigercontrol_dir, "problems/control/pybullet/URDFs/plane.urdf"))

        # Load robot from URDF
        self.husky = pybullet.loadURDF(os.path.join(tigercontrol_dir, "problems/control/pybullet/URDFs/husky.urdf"), globalScaling=0.5)

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
        self.obsUid = self.generate_obstacles(pybullet, heightObs, robotRadius)

        # Initialize position of robot
        self.state = [0.0, 1.0, 0.0] # [x, y, theta]
        quat = pybullet.getQuaternionFromEuler([0.0, 0.0, self.state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2

        pybullet.resetBasePositionAndOrientation(self.husky, [self.state[0], self.state[1], 0.0], quat)
        pybullet.resetBasePositionAndOrientation(self.sphere, [self.state[0], self.state[1], self.robotHeight], [0,0,0,1])

        return (np.array(self.state), params, self.husky, self.sphere, self.obsUid)

    def render(self, mode='human', close=False):
        return

    # Helper functions for setting up obstacle environments, simulating dynamics, etc.
    # Parameters
    def default_params(self):
        
        params = {} # Initialize parameter dictionary
        params['numRays'] = 401 # number of rays for sensor measurements
        params['senseRadius'] = 6.0 # sensing radius
        params['robotRadius'] = 0.27 # radius of robot
        params['robotHeight'] = 0.15/2 # rough height of COM of robot
        params['th_min'] = -np.pi/2 # sensing angle minimum 
        params['th_max'] = np.pi/2 # sensing angle maximum
        params['T_horizon'] = 200 # time horizon over which to evaluate everything   
        
        # precompute vector of angles for sensor
        params['thetas_nominal'] = np.reshape(np.linspace(params['th_min'], params['th_max'], params['numRays']), (params['numRays'],1))
        
        return params 

    # Create some obstacles 
    def generate_obstacles(self, p, heightObs, robotRadius, numObs=45, x_lim=[-5.0, 5.0], y_lim=[0.0, 24.0]):
        '''
        Params:
            p : pybullet
            heightObs: height of cylinders
            robotRadius: radius of robot
            numObs: number of cylinders to generate
            x_lim (list): [lower, upper] box for x-coordinate
            y_lim (list): [lower, upper] box for y-coordinate

        '''
        
        # First create bounding obstacles
        # x_lim = [-5.0, 5.0]
        # y_lim = [0.0, 24.0]
            
        #numObs = 10 + np.random.randint(0,5)
        # NUMBER OF OBJECTS
        # numObs = 45

        # radiusObs = 0.15
        massObs = 0
        visualShapeId = -1
        linkMasses = [None]*(numObs+3) # +3 is because we have three bounding walls
        colIdxs = [None]*(numObs+3)
        visIdxs = [None]*(numObs+3)
        posObs = [None]*(numObs+3)
        orientObs = [None]*(numObs+3)
        parentIdxs = [None]*(numObs+3)
        
        linkInertialFramePositions = [None]*(numObs+3)
        linkInertialFrameOrientations = [None]*(numObs+3)
        linkJointTypes = [None]*(numObs+3)
        linkJointAxis = [None]*(numObs+3)

        for obs in range(numObs):
            
            linkMasses[obs] = 0.0
            visIdxs[obs] = -1 # p.createVisualShape(p.GEOM_CYLINDER,radiusObs,[1,1,1],heightObs,rgbaColor=[0,0,0,1])
            parentIdxs[obs] = 0
        
            linkInertialFramePositions[obs] = [0,0,0]
            linkInertialFrameOrientations[obs] = [0,0,0,1]
            linkJointTypes[obs] = p.JOINT_FIXED
            linkJointAxis[obs] = np.array([0,0,1]) # [None]*numObs
            
            posObs_obs = np.array([None]*3)
            posObs_obs[0] = x_lim[0] + (x_lim[1] - x_lim[0])*np.random.random_sample(1) 
            posObs_obs[1] = 4.0 + y_lim[0] + (y_lim[1] - y_lim[0] - 2.0)*np.random.random_sample(1) # Push up a bit (by 4.0)
            posObs_obs[2] = 0 # set z at ground level
            posObs[obs] = posObs_obs # .tolist()
            orientObs[obs] = [0,0,0,1]
            colIdxs[obs] = p.createCollisionShape(p.GEOM_CYLINDER,radius=(0.2)*np.random.random_sample(1)+0.05,height=heightObs)
            # colIdxs[obs] = p.createCollisionShape(p.GEOM_CYLINDER,radius=radiusObs,height=heightObs)

        # Create bounding objects
        # Left wall
        linkMasses[numObs] = 0.0
        visIdxs[numObs] = -1 # p.createVisualShape(p.GEOM_BOX, halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2], rgbaColor=[0.8,0.1,0.1,1.0]) # -1
        parentIdxs[numObs] = 0
        linkInertialFramePositions[numObs] = [0,0,0]
        linkInertialFrameOrientations[numObs] = [0,0,0,1]
        linkJointTypes[numObs] = p.JOINT_FIXED
        linkJointAxis[numObs] = np.array([0,0,1]) 
        posObs[numObs] = [x_lim[0], (y_lim[0]+y_lim[1])/2.0, 0.0]
        orientObs[numObs] = [0,0,0,1]
        colIdxs[numObs] = p.createCollisionShape(p.GEOM_BOX, halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2])
        
        
        # Right wall
        linkMasses[numObs+1] = 0.0
        visIdxs[numObs+1] = -1 # p.createVisualShape(p.GEOM_BOX, halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2], rgbaColor=[0.8,0.1,0.1,1.0]) # -1
        parentIdxs[numObs+1] = 0
        linkInertialFramePositions[numObs+1] = [0,0,0]
        linkInertialFrameOrientations[numObs+1] = [0,0,0,1]
        linkJointTypes[numObs+1] = p.JOINT_FIXED
        linkJointAxis[numObs+1] = np.array([0,0,1]) 
        posObs[numObs+1] = [x_lim[1], (y_lim[0]+y_lim[1])/2.0, 0.0]
        orientObs[numObs+1] = [0,0,0,1]
        colIdxs[numObs+1] = p.createCollisionShape(p.GEOM_BOX, halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2])
        
        # Bottom wall
        linkMasses[numObs+2] = 0.0
        visIdxs[numObs+2] = -1 # p.createVisualShape(p.GEOM_BOX, halfExtents = [0.1, (x_lim[1] - x_lim[0])/2.0, heightObs/2], rgbaColor=[0.8,0.1,0.1,1.0])
        parentIdxs[numObs+2] = 0
        linkInertialFramePositions[numObs+2] = [0,0,0]
        linkInertialFrameOrientations[numObs+2] = [0,0,0,1]
        linkJointTypes[numObs+2] = p.JOINT_FIXED
        linkJointAxis[numObs+2] = np.array([0,0,1]) 
        posObs[numObs+2] = [(x_lim[0]+x_lim[1])/2.0, y_lim[0], 0.0]
        orientObs[numObs+2] = [0,0,np.sqrt(2)/2,np.sqrt(2)/2]
        colIdxs[numObs+2] = p.createCollisionShape(p.GEOM_BOX, halfExtents = [0.1, (x_lim[1] - x_lim[0])/2.0, heightObs/2])        
            
        obsUid = p.createMultiBody(baseCollisionShapeIndex = -1, 
                                   baseVisualShapeIndex = -1, 
                                   basePosition = [0,0,0], 
                                   baseOrientation = [0,0,0,1], 
                                   baseInertialFramePosition = [0,0,0], 
                                   baseInertialFrameOrientation = [0,0,0,1], 
                                   linkMasses = linkMasses, 
                                   linkCollisionShapeIndices = colIdxs, 
                                   linkVisualShapeIndices = visIdxs, 
                                   linkPositions = posObs, 
                                   linkOrientations = orientObs, 
                                   linkParentIndices = parentIdxs, 
                                   linkInertialFramePositions = linkInertialFramePositions, 
                                   linkInertialFrameOrientations = linkInertialFrameOrientations, 
                                   linkJointTypes = linkJointTypes, 
                                   linkJointAxis = linkJointAxis)
        
        return obsUid

    # Simulate range sensor (get distances along rays)        
    def getDistances(self, p): 

        # Get distances 
        # rays emanate from robot
        state = self.state
        robotHeight = self.params['robotHeight']
        numRays = self.params['numRays']
        senseRadius = self.params['senseRadius']
        thetas_nominal = self.params['thetas_nominal']

        raysFrom = np.concatenate((state[0]*np.ones((numRays,1)), state[1]*np.ones((numRays,1)), robotHeight*np.ones((numRays,1))), 1)

        thetas = (-state[2]) + thetas_nominal # Note the minus sign: +ve direction for state[2] is anti-clockwise (right hand rule), but sensor rays go clockwise

        raysTo = np.concatenate((state[0]+senseRadius*np.sin(thetas), state[1]+senseRadius*np.cos(thetas), robotHeight*np.ones((numRays,1))), 1)

        coll = p.rayTestBatch(raysFrom, raysTo)

        dists = np.zeros((1,numRays))
        for i in range(numRays):
            dists[0][i] = senseRadius*coll[i][2]
                
        return dists

class Obstacles(PyBulletProblem):
    """
    Description: Simulates a obstacles avoidance environment
    """
    compatibles = set(['Obstacles-v0', 'PyBullet'])

    def __init__(self):
        self.initialized = False

    def initialize(self, render=False, params=None):
        self.initialized = True
        self._env = ObstaclesEnv(renders=render, params=params)
        self.observation_space = self._env.observation_space.shape
        self.action_space = self._env.action_space.shape
        self.state = None
        state, params, husky, sphere, obsUid = self._env.reset()
        return (state, params, husky, sphere, obsUid)

    def step(self, action):
        return self._env.step(action)

    def step_fgm(self, angle):
        return self._env.step_fgm(angle)

    def render(self, mode='human', close=False):
        self._env.render(mode=mode, close=close)

    def getDistances(self, p):
        return self._env.getDistances(p)






