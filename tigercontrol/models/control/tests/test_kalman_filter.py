# test the KalmanFilter model class

import tigercontrol
import jax.numpy as np
import jax.random as random
from tigercontrol.utils import generate_key
import matplotlib.pyplot as plt

# Test Kalman Filter for constant signal x = 0.5 with measurement noise 0.1
def test_kalman_filter(steps=100, show_plot=True):
    T = steps 

    x_true = 1
    env_noise = 0.3
    x0 = 0

    model = tigercontrol.model("KalmanFilter")
    model.initialize(x0, 1, 0, 1, 1, 0, env_noise)

    loss = lambda x_true, x_pred: (x_true - x_pred)**2

    results = []
    for i in range(T):
        z = x_true + float(random.normal(generate_key(), shape = (1,)) * env_noise)
        x_pred = model.step(0, z)
        cur_loss = float(loss(x_true, x_pred))
        results.append(cur_loss)

    if show_plot:
        plt.plot(results)
        plt.title("KalmanFilter model on constant signal")
        plt.show(block=False)
        plt.pause(15)
        plt.close()
    print("test_kalman_filter passed")
    return

if __name__=="__main__":
    test_kalman_filter()