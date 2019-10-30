# test the CustomController class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt

# test a simple CustomController that returns last value by storing a single param
def test_custom_controller(steps=1000, show_plot=True):
    # initial preparation
    T = steps 
    p, q = 3, 3
    loss = lambda y_true, y_pred: (y_true - y_pred)**2
    environment = tigercontrol.environment("ARMA-v0")
    cur_x = environment.initialize(p, q)

    # simple LastValue custom controller implementation
    class Custom(tigercontrol.CustomController):
        def initialize(self):
            self.x = 0.0
        def predict(self, x):
            self.x = x
            return self.x
        def update(self, y):
            pass

    # try registering and calling the custom controller
    tigercontrol.register_custom_controller(Custom, "TestCustomController")
    custom_controller = tigercontrol.controllers("TestCustomController")
    custom_controller.initialize()

    # regular LastValue controller as sanity check
    reg_controller = tigercontrol.controllers("LastValue")
    reg_controller.initialize()
 
    results = []
    for i in range(T):
        cur_y_pred = custom_controller.predict(cur_x)
        reg_y_pred = reg_controller.predict(cur_x)
        assert cur_y_pred == reg_y_pred # check that CustomController outputs the correct thing
        cur_y_true = environment.step()
        custom_controller.update(cur_y_true)
        reg_controller.update(cur_y_true)
        results.append(loss(cur_y_true, cur_y_pred))
        cur_x = cur_y_true

    if show_plot:
        plt.plot(results)
        plt.title("Custom (last value) controller on ARMA environment")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_custom_controller passed")
    return

if __name__=="__main__":
    test_custom_controller()

