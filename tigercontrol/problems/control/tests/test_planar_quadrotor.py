"""
Test for PlanarQuadrotor problem
"""
import time
import tigercontrol
import jax
import jax.numpy as np
import matplotlib.pyplot as plt

def test_planar_quadrotor(steps=10, show_plot=True):
    T = steps
    n = 6 # number of states
    m = 2 # number of control inputs
    d = n
    # n = 2 # input/control dimension
    # m = 6 # output dimension
    # d = 6 # hidden/state dimension (which is equal to output dimension here)
    problem = tigercontrol.problem("PlanarQuadrotor-v0")
    problem.initialize() # (m, d, n)

    # Define nominal operating point (hover)
    x0 = np.zeros(n)
    u0 = (problem.m*problem.g/2.0)*np.ones(2) # control inputs required to cancel gravity

    # Get linearized dynamics
    A, B = problem.linearize_dynamics(x0, u0)

    f = np.zeros((n,1)) # bias
    C = np.identity(n+m) # quadratic cost

    c = np.zeros((n+m, 1)) # linear cost
    x = problem.initialize() # np.ones((n,1)) # initial state

    method = tigercontrol.method("LQR")
    method.initialize(A, B, C, T, x)

    u = method.plan()

    if show_plot:
        plt.plot([np.linalg.norm(ui) for ui in u], 'C0--', label = "LQR")
        plt.title("LQR (open loop control inputs) on PlanarQuadrotor")
        plt.legend()
        plt.show(block=False)
        plt.pause(10)
        plt.close()

    # TODO: Once control library LQR supports closed-loop control, plot the trajectory
    # trajectory = []
    # for i in range(steps):
    #     # Compute control input
    #     u = u0 + method.plan(x) # nominal control input + LQR feedback: u0 + K*(x-x0)
    #     # Apply control input
    #     x = problem.step(u)
    #     # Append to trajectory
    #     trajectory.append(x)
 
    
    # if show_plot:
    #     plt.plot([np.linalg.norm(xi-x0) for xi in trajectory], 'C0--', label = "LQR")
    #     plt.title("LQR on PlanarQuadrotor")
    #     plt.legend()
    #     plt.show(block=False)
    #     plt.pause(10)
    #     plt.close()

    problem.close()
    print("test_planar_quadrotor passed")


if __name__ == "__main__":
    test_planar_quadrotor()