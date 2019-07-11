# Test the ODEShootingMethod model class by solving x'' = x + 4exp(t) for L = 1

import ctsb
import jax.numpy as np
import math
import matplotlib.pyplot as plt

def y(t):
    return math.exp(t) * (1 + 2 * t)

def f(y, t):
    return y + 4 * math.exp(t)

def test_ode_shooting_method(steps=10, show_plot=True):
    T = steps 
    L = 1
    
    t = L / 2
    y_true = y(t)

    model = ctsb.model("ODEShootingMethod")
    model.initialize(f, y(0), y(L), 3.0, 4.0, t)

    loss = lambda x_true, x_pred: (x_true - x_pred)**2

    results = []

    for i in range(T):
        y_pred = model.step()
        cur_loss = float(loss(y_true, y_pred))
        results.append(cur_loss)

    if show_plot:
        plt.plot(results)
        plt.title("ODEShootingMethod")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    print("test_ode_shooting_method passed")
    return

if __name__=="__main__":
    test_ode_shooting_method()