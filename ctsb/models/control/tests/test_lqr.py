# test the LQR model class

import tigercontrol
import jax.numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def test_lqr(steps=10, show_plot=True):

    model = tigercontrol.model("LQR")
    lambda1 = 0.5
    lambda2 = 10

    # Test functionality for floats
    F = np.reshape(np.ones(2), (1, 2))
    f = 0
    C = np.array([[1, 0], [0, lambda1]])
    c = 0
    T = steps 
    x = 1

    model.initialize(F, f, C, c, T, x)

    u = model.plan()

    # Plot control found
    if show_plot:
        plt.plot([np.linalg.norm(i) for i in u], label = "LQR on floats small lambda")

    C = np.array([[1, 0], [0, lambda2]])

    model.initialize(F, f, C, c, T, x)

    u = model.plan()

    # Plot control found
    if show_plot:
        plt.plot([np.linalg.norm(i) for i in u], 'C0--', label = "LQR on floats big lambda")

    # Test functionality for higher dimensions
    lambda1 = 0.5
    lambda2 = 100

    xdim, udim = 2, 3

    F = np.ones((xdim, xdim + udim))
    f = np.zeros((xdim, 1))
    C = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, lambda1, 0, 0], [0, 0, 0, lambda1, 0], [0, 0, 0, 0, lambda1]])
    c = np.zeros((xdim + udim, 1))
    T = steps
    x = np.ones((xdim, 1))

    model.initialize(F, f, C, c, T, x)

    u = model.plan()

    # Plot control found
    if show_plot:
        plt.plot([np.linalg.norm(i) for i in u], label = "LQR on multiple dim small lambda")

    C = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, lambda2, 0, 0], [0, 0, 0, lambda2, 0], [0, 0, 0, 0, lambda2]])

    model.initialize(F, f, C, c, T, x)

    u = model.plan()

    # Plot control found
    if show_plot:
        plt.plot([np.linalg.norm(i) for i in u], 'C1--', label = "LQR on multiple dim big lambda")

    if show_plot:
        plt.title("LQR")
        plt.legend()
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    print("test_lqr passed")
    return

if __name__=="__main__":
    test_lqr()