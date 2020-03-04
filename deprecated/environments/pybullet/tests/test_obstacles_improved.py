"""
Test for PyBullet cartpole environment
"""
import time
import tigercontrol
import numpy as np
import jax.random as random
from tigercontrol.utils import generate_key
import pybullet as pybullet
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

def compute_fgm_angle(y_bad, thetas_nominal_bad, state):
    print(" ---------------- compute_fgm_angle ----------------")

    # fix bad list
    y = 1.0 / np.flip(y_bad[0])
    thetas_nominal = [t[0] for t in thetas_nominal_bad]
    x_coord, y_coord, curr_angle = state[0], state[1], state[2]

    car_width = 1.2 * (2 * 0.27) # safety threshold * width of car
    d_min = 3.0
    threshold = 1.0 / d_min # inverse of distance
    theta_delta = (2.0 * np.pi / 3) / 401 # 401 rays of observation
    theta_min = 2 * np.tan(car_width / (2 * d_min))
    ind_min_width = 2.0 * int(np.ceil(theta_min / theta_delta)) # increase min width
    half_ind_min = int(ind_min_width / 2)

    valid_gaps = []
    gap_costs = []
    left, curr_gap = 0, 0
    for i in range(half_ind_min, len(y) - half_ind_min):
        mean_cost = np.mean(y[i-half_ind_min:i+half_ind_min])
        max_cost = np.mean(y[i-half_ind_min:i+half_ind_min])
        cost = max_cost
        if cost < threshold:
            valid_gaps.append(i)
            gap_costs.append(cost)

    if valid_gaps == []:
        return 0.95 * curr_angle # slowly moves toward center

    target_angle = x_coord / 10.0 # 10 is width of field
    angles = [thetas_nominal[i] for i in valid_gaps]
    angle_diff = [np.abs(ang - target_angle) for ang in angles]
    curr_angle_diff = [np.abs(ang - curr_angle) for ang in angles]

    a,b,c = 10 / np.mean(gap_costs), 2 / np.mean(angle_diff), 4 / np.mean(curr_angle_diff) # learned parameter
    total_cost = [a * d + b * t + c * p for d,t,p in zip(gap_costs, angle_diff, curr_angle_diff)]

    opt_angle = angles[np.argmin(np.array(total_cost))]
    print("\ncosts: " + str(gap_costs))
    print("\naction: " + str(opt_angle))
    return opt_angle


# Precompute costs for different environments and controllers
def precompute_environment_costs(numEnvs, K, L, params, husky, sphere, GUI, seed, environment, obsUid):
    
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
                y = environment.getDistances(pybullet)

                # Compute control input
                # u = compute_control(y, K[l])
                angle = compute_fgm_angle(y, thetas_nominal, state)
                all_angles.append(angle)

                # Update state
                # state = robot_update_state(state, u)
                state, cost_env_l, done, _ = environment.step_fgm(angle)

                if cost_env_l == 1.0:
                    print(state)
                    print("\nwoops\n")

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
    environment = tigercontrol.environment("PyBullet-Obstacles-v0")
    # obs = environment.initialize(render=verbose)

    # controller = tigercontrol.controllers("CartPoleNN")
    # controller.initialize(environment.get_observation_space(), environment.get_action_space())

    # Initial setup
    # Flag that sets if things are visualized
    # GUI = True; # Only for debugging purposes

    GUI = True
    random_seed =5 
    m = 1
    state, params, husky, sphere, obsUid = environment.initialize(render=verbose)
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

    costs_precomputed = precompute_environment_costs(m, K, L, params, husky, sphere, GUI, random_seed, environment, obsUid)

    print("costs_precomputed: " + str(costs_precomputed))
    print("test_obstacles passed")


if __name__ == "__main__":
    #test_obstacles(verbose=True)
    pass
