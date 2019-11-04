"""
Test for PyBullet pendulum environment
"""
import time
import tigercontrol
import jax.numpy as np


# cartpole test
def test_pendulum(verbose=False):
    environment = tigercontrol.environment("Pendulum-v0")
    #L = lambda x, u: 1. - x[1] # 1 - cos(theta), where theta=0 is the goal (pendulum pointing up)
    # L = lambda x, u: x[0]**2

    # C_x, C_u = np.diag(np.array([0.1, 0.0, 0.0, 0.0])), np.diag(np.array([0.1]))
    # L = lambda x, u: x.T @ C_x @ x + u.T @ C_u @ u

    dim_x, dim_u = 2, 1
    obs = environment.initialize()

    
    T = 50 # horizon
    threshold = 0.01
    lamb = 0.1
    max_iterations = 50

    controller = tigercontrol.controller("ILQR")
    controller.initialize(environment, dim_x, dim_u, max_iterations, lamb, threshold)

    if verbose:
        print("Running iLQR...")
    u = controller.plan(obs, T)

    total_cost = 0
    #print("u: " + str([float(u_t) for u_t in u]))
    index = 0
    for t in range(25 * T):
        # print("t:" + str(t))
        if verbose: 
            environment.render()
            time.sleep(1. / 30.)
        obs, cost, done = environment.step(u[index])
        total_cost += cost
        index += 1

        if done:
            if verbose:
                print("lasted {} time steps".format(t+1))
            obs = environment.initialize()
        
        if done or index == T:
            if verbose:
                print("recomputing u...")
            u = controller.plan(obs, T)
            print(total_cost)
            total_cost = 0
            index = 0

    environment.close()
    print("test_pendulum passed")


if __name__ == "__main__":
    test_pendulum(verbose=True)
