# test the CustomModel class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt

# test a simple CustomModel that returns last value by storing a single param
def test_custom_model(steps=1000, show_plot=True):
    # initial preparation
    T = steps 
    p, q = 3, 3
    loss = lambda y_true, y_pred: (y_true - y_pred)**2
    problem = tigercontrol.problem("ARMA-v0")
    cur_x = problem.initialize(p, q)

    # simple LastValue custom model implementation
    class Custom(tigercontrol.CustomModel):
        def initialize(self):
            self.x = 0.0
        def predict(self, x):
            return self.x
        def update(self, y):
            self.x = y

    # try registering and calling the custom model
    tigercontrol.register_custom_model(Custom, "TestCustomModel")
    custom_model = tigercontrol.model("TestCustomModel")
    custom_model.initialize()

    # regular LastValue model as sanity check
    reg_model = tigercontrol.model("LastValue")
    reg_model.initialize()
 
    results = []
    for i in range(T):
        cur_y_pred = custom_model.predict(cur_x)
        reg_y_pred = reg_model.predict(cur_x)
        assert cur_y_pred == reg_y_pred # check that CustomModel outputs the correct thing
        cur_y_true = problem.step()
        custom_model.update(cur_y_true)
        reg_model.update(cur_y_true)
        results.append(loss(cur_y_true, cur_y_pred))
        cur_x = cur_y_true

    if show_plot:
        plt.plot(results)
        plt.title("Custom (last value) model on ARMA problem")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_custom_model passed")
    return

if __name__=="__main__":
    test_custom_model()

