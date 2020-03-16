# test the CustomController class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt

# test a simple CustomController that returns last value by storing a single param
def test_custom_controller(steps=1000, show_plot=False):
    # initial preparation
    T = steps 
    loss = lambda y_true, y_pred: (y_true - y_pred)**2
    environment = tigercontrol.environment("LDS") # return class
    n, m = 2, 2 # state and action dimension
    env = environment()
    cur_x = env.initialize(n, m)

    # simple zero custom controller implementation
    class Custom(tigercontrol.CustomController):
        def get_action(self, x):
            return np.zeros((m,1))
        def update(self, y):
            pass

    # try registering and calling the custom controller
    tigercontrol.register_custom_controller(Custom, "TestCustomController")
    custom_controller = tigercontrol.controller("TestCustomController")
    controller = custom_controller()
 
    results = []
    for i in range(T):
        a = controller.get_action(cur_x)
        cur_x = env.step(a)

    if show_plot:
        plt.plot(results)
        plt.title("Custom (last value) controller on LQR environment")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_custom_controller passed")
    return

if __name__=="__main__":
    test_custom_controller()

