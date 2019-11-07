"""
Test for PyBullet pendulum environment
"""
import time
import tigercontrol
from tigercontrol.controllers.ilqr_pendulum_3_state import ILQR as ilqr_3_state
import jax.numpy as np


# cartpole test
def test_pendulum(verbose=False):
    environment = tigercontrol.environment("Pendulum3State-v0")
    #L = lambda x, u: 1. - x[1] # 1 - cos(theta), where theta=0 is the goal (pendulum pointing up)
    # L = lambda x, u: x[0]**2

    # C_x, C_u = np.diag(np.array([0.1, 0.0, 0.0, 0.0])), np.diag(np.array([0.1]))
    # L = lambda x, u: x.T @ C_x @ x + u.T @ C_u @ u

    dim_x, dim_u = 3, 1
    obs = environment.initialize()


    
    T = 300 # horizon
    threshold = 0.000001
    lamb = 1.0
    max_iterations = 200

    controller = ilqr_3_state()
    controller.initialize(environment, dim_x, dim_u, max_iterations, lamb, threshold)
    print("initial obs:" + str(controller.reduce_state(obs)))

    if verbose:
        print("Running iLQR...")
    u, pos_0, u_0 = controller.plan(obs, T)
    print(u)
    pos = 0
    for i in range(len(u)):
        if u[i][0] > 0:
            pos += 1
    print("pos = " + str(pos))
    print("neg = " + str(T-pos))
    print("pos_0 = " + str(pos_0))
    print("neg_0 = " + str(T-pos_0))
    print("u_norm = " + str(np.linalg.norm(u)))
    print("u_0_norm = " + str(np.linalg.norm(u_0)))
    print("sum(u) = " + str(np.sum(u)))
    print("sum(u_0) = " + str(np.sum(u_0)))

    total_cost = 0
    #print("u: " + str([float(u_t) for u_t in u]))
    index = 0
    for t in range(10*T):
        # print("---------------------------------------")
        # print("t:" + str(t))
        # print("obs: " + str(obs))
        if verbose: 
            environment.render()
            time.sleep(1. / 30.)
        obs, cost, done = environment.step(u[index])
        # total_cost += cost
        index += 1
        '''
        if done:
            if verbose:
                print("lasted {} time steps".format(t+1))
            obs = environment.initialize()
        '''
        if done or index == T:
            if verbose:
                print("recomputing u...")

                # print(total_cost)
            environment.close()
            obs = environment.initialize()
            print("initial obs:" + str(controller.reduce_state(obs)))

            controller.initialize(environment, dim_x, dim_u, max_iterations, lamb, threshold)
            u, pos_0, u_0 = controller.plan(obs, T)
            pos = 0
            for i in range(len(u)):
                if u[i][0] > 0:
                    pos += 1
            print(u)
            print("pos = " + str(pos))
            print("neg = " + str(T-pos))
            print("pos_0 = " + str(pos_0))
            print("neg_0 = " + str(T-pos_0))
            print("u_norm_2 = " + str(np.linalg.norm(u)))
            print("u_0_norm_2 = " + str(np.linalg.norm(u_0)))
            print("sum(u) = " + str(np.sum(u)))
            print("sum(u_0) = " + str(np.sum(u_0)))

            total_cost = 0
            index = 0

    environment.close()
    print("test_pendulum passed")


if __name__ == "__main__":
    test_pendulum(verbose=True)
