# test the CustomProblem class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt

# test a simple CustomProblem that returns last value by storing a single param
def test_custom_problem(steps=1000, show=True):
    # initial preparation
    T = steps
    loss = lambda y_true, y_pred: (y_true - y_pred)**2
    model = tigercontrol.model("LastValue")
    model.initialize()

    # simple custom Problem that returns alternating +/- 1.0
    class Custom(tigercontrol.CustomProblem):
        def initialize(self):
            self.T = 0
            return -1
        def step(self):
            self.T += 1
            return 2 * (self.T % 2) - 1

    # try registering and calling the custom problem
    tigercontrol.register_custom_problem(Custom, "TestCustomProblem")
    custom_problem = tigercontrol.problem("TestCustomProblem")
    cur_x = custom_problem.initialize()
 
    results = []
    for i in range(T):
        cur_y_pred = model.predict(cur_x)
        cur_y_true = custom_problem.step()
        results.append(loss(cur_y_true, cur_y_pred))
        cur_x = cur_y_true

    if show:
        plt.plot(results)
        plt.title("LastValue model on custom alternating problem")
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    print("test_custom_problem passed")
    return

if __name__=="__main__":
    test_custom_problem()

