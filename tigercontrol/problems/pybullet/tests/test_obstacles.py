"""
Test for PyBullet cartpole problem
"""
import time
import tigercontrol
import jax.numpy as np
import jax.random as random
from tigercontrol.utils import generate_key
import pybullet as pybullet
from tigercontrol.problems.pybullet.obstacle_utils import *
import os.path
from os import path
from pathlib import Path

# Compute control input    
def compute_control(y, K):
    # y is vector of depth measurements
    print("y : " + str(y))
    print("K: " + str(K))
    print("1/y: " + str(1./y))
    u_diff = np.matmul(1./y, K)
    print("u_diff: " + str(u_diff))
       
    return u_diff

def compute_fgm_angle(y, thetas_nominal, cur_angle):
    print(" ---------------- compute_fgm_angle ----------------")
    threshold = 5.0 # 0.7 * np.max(y[0])
    y_filter = [x >= threshold for x in y[0]] + [False]
    run, best_run = 0, 0
    left, right = None, None
    best_gaps = []
    for i in range(len(y_filter)):
        if y_filter[i]:
            if run == 0:
                left = i
            run += 1
        else:
            if run > best_run:
                right = i - 1
                best_run = run
                best_gaps = [(left, right)]
            elif run == best_run and best_run > 0:
                right = i - 1
                best_gaps.append((left, right))
            run = 0

    avg_depth_mapper = lambda l_r: np.mean([x for x in y[0][range(l_r[0], l_r[1]+1)]])
    avg_depth = list(map(avg_depth_mapper, best_gaps))
    angle_mapper = lambda l_r : (thetas_nominal[l_r[0]][0] + thetas_nominal[l_r[1]][0])/2.0
    best_angles = list(map(angle_mapper, best_gaps))

    print("best_angles : " + str(best_angles))
    print("avg_depth : " + str(avg_depth))
    
    opt_angle = best_angles[np.argmax(avg_depth)]

    print("opt_angle: " + str(opt_angle))
    return opt_angle


# Precompute costs for different environments and controllers
def precompute_environment_costs(numEnvs, K, L, params, husky, sphere, GUI, seed, problem, obsUid):
    
    # Parameters
    numRays = params['numRays']
    senseRadius = params['senseRadius']   
    robotRadius = params['robotRadius']
    robotHeight = params['robotHeight']
    thetas_nominal = params['thetas_nominal']
    T_horizon = params['T_horizon']
    
    # Fix random seed for consistency of results
    np.random.seed(seed)
    
    # Initialize costs for the different environments and different controllers
    costs = np.zeros((numEnvs, L))
    
    for env in range(0,numEnvs):
                
        # Print
        if (env%10 == 0):
            print(env, "out of", numEnvs)
    
        # Sample environment
        # heightObs = 20*robotHeight
        # obsUid = generate_obstacles(pybullet, heightObs, robotRadius)  

        for l in range(0,L):
            
            # Initialize position of robot
            state = [0.0, 1.0, 0.0] # [x, y, theta]
            quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2

            pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
            pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])

            # Cost for this particular controller (lth controller) in this environment
            cost_env_l = 0.0
            all_angles = []

            for t in range(0, T_horizon):

                # Get sensor measurement
                y = getDistances(pybullet, state, robotHeight, numRays, senseRadius, thetas_nominal)

                # Compute control input
                # u = compute_control(y, K[l])
                angle = compute_fgm_angle(y, thetas_nominal, state[2])
                all_angles.append(angle)

                # Update state
                # state = robot_update_state(state, u)
                state, cost_env_l, done, _ = problem.step_fgm(angle)

                # Update position of pybullet object
                # quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2
                # pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
                # pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])    

                if (GUI):
                    pybullet.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[state[0], state[1], 2*robotHeight])

                    time.sleep(0.025) 


                # Check if the robot is in collision. If so, cost = 1.0.      
                # Get closest points. Note: Last argument is distance threshold. Since it's set to 0, the function will only return points if the distance is less than zero. So, closestPoints is non-empty iff there is a collision.
                # closestPoints = pybullet.getClosestPoints(sphere, obsUid, 0.0)


                # See if the robot is in collision. If so, cost = 1.0. 
                '''
                if closestPoints: # Check if closestPoints is non-empty 
                    cost_env_l = 1.0
                    break # break out of simulation for this environment
                '''
                if cost_env_l == 1.0:
                    break;
            
            # Check that cost is between 0 and 1 (for sanity)
            if (cost_env_l > 1.0):
                raise ValueError("Cost is greater than 1!")
                
            if (cost_env_l < 0.0):
                raise ValueError("Cost is less than 0!")
            
            # Record cost for this environment and this controller
            costs[env][l] = cost_env_l
            
        # Remove obstacles
        pybullet.removeBody(obsUid)
        print("mean angle = " + str(sum(all_angles)/len(all_angles)))
    
    return costs

# cartpole test
def test_obstacles(verbose=False):
    problem = tigercontrol.problem("PyBullet-Obstacles-v0")
    # obs = problem.initialize(render=verbose)

    # model = tigercontrol.model("CartPoleNN")
    # model.initialize(problem.get_observation_space(), problem.get_action_space())

    # Initial setup
    # Flag that sets if things are visualized
    # GUI = True; # Only for debugging purposes

    GUI = True
    random_seed =5 
    m = 1
    state, params, husky, sphere, obsUid = problem.initialize(render=verbose)
    numRays = params['numRays']
    thetas_nominal = params['thetas_nominal']

    # Controller and optimization setup

    # Choose L controllers
    num_x_intercepts = 1
    num_y_intercepts = 1
    L = num_x_intercepts*num_y_intercepts

    x_intercepts = np.linspace(0.1, 5.0, num_x_intercepts)
    y_intercepts = np.linspace(0.0, 10.0, num_y_intercepts)

    print("x_intercepts: " + str(x_intercepts))
    print("y_intercepts: " + str(y_intercepts))
    print("thetas_nominal: " + str(thetas_nominal))
    print("numRays: " + str(numRays))

    K = L*[None]
    for i in range(num_x_intercepts):
        for j in range(num_y_intercepts):
            K[i*num_y_intercepts + j] = np.zeros((numRays,1))
            for r in range(numRays):
                if (thetas_nominal[r] > 0):
                    K[i*num_y_intercepts + j][r] = y_intercepts[j]*(x_intercepts[i] - thetas_nominal[r])/x_intercepts[i]
                else:
                    K[i*num_y_intercepts + j][r] = y_intercepts[j]*(-x_intercepts[i] - thetas_nominal[r])/x_intercepts[i]

    print("First K = " + str(K))

    costs_precomputed = precompute_environment_costs(m, K, L, params, husky, sphere, GUI, random_seed, problem, obsUid)

    print("costs_precomputed: " + str(costs_precomputed))
    print("test_obstacles passed")


if __name__ == "__main__":
    test_obstacles(verbose=True)

