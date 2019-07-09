# test the LQR model class

import ctsb
import jax.numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def test_lqr(steps=10, show_plot=True):

    model = ctsb.model("LQR")
    reg_ct = 0.5

    # Test functionality for floats
    F = np.reshape(np.ones(2), (1, 2))
    f = 0
    C = np.array([[1, 0], [0, reg_ct]])
    c = 0
    T = steps 
    x = 1

    model.initialize(F, f, C, c, T, x)

    u = model.step()

    # Plot control found
    if show_plot:
        plt.plot([float(i) for i in u], label = "LQR on floats")

    # Test functionality for higher dimensions
    xdim, udim = 2, 3

    F = np.ones((xdim, xdim + udim))
    f = np.zeros((xdim, 1))
    C = np.eye(xdim + udim)
    c = np.zeros((xdim + udim, 1))
    T = steps
    x = np.ones((xdim, 1))

    model.initialize(F, f, C, c, T, x)

    u = model.step()

    # Plot control found
    if show_plot:
        plt.plot([(float(i[0]), float(i[1]), float(i[2])) for i in u], label = "LQR on multiple dimensions")
        plt.title("LQR")
        plt.legend()
        plt.show()
        plt.close()

    print("test_lqr passed")
    return

if __name__=="__main__":
    test_lqr()