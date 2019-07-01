# test the CusomModel class

import ctsb
import jax.numpy as np
import matplotlib.pyplot as plt
from ctsb.models.custom_model import CustomModel

# test a simple CustomModel that returns last value by storing a single param
def test_custom_model(steps=100, show_plot=True):
    T = steps 
    p, q = 3, 3

    predict = lambda params, x: x
    update = lambda params, x: params
    params = 0.0
    custom_model = CustomModel()
    custom_model.initialize(predict=predict, params=params, update=update)
    reg_model = ctsb.model("LastValue")
    reg_model.initialize() # sanity check

    loss = lambda y_true, y_pred: (y_true - y_pred)**2
    problem = ctsb.problem("ARMA-v0")
    cur_x = problem.initialize(p, q)
 
    results = []
    for i in range(T):
        cur_y_pred = custom_model.predict(cur_x)
        reg_y_pred = reg_model.predict(cur_x)
        assert cur_y_pred == reg_y_pred # check that CustomModel outputs the correct thing
        cur_y_true = problem.step()
        cur_loss = loss(cur_y_true, cur_y_pred)
        results.append(cur_loss)
        custom_model.update(cur_y_true)
        cur_x = cur_y_true

    assert custom_model.get_params() == 0.0 # check that params haven't changed

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