""" 
test SimpleBoost model
note: n=3 seems to do consistently better than other n's...
"""
import jax.numpy as np
import ctsb
from ctsb.models.optimizers import OGD
from ctsb.models.optimizers.losses import mse
import matplotlib.pyplot as plt

def test_simple_boost(steps=1000, show=False):
    test_simple_boost_arma(steps=steps, show=show)
    test_simple_boost_lstm(steps=steps, show=show)
    print("test_simple_boost passed")

def avg_regret(loss):
    avg_regret = []
    cur_avg = 0
    for i in range(len(loss)):
        cur_avg = (i / (i + 1)) * cur_avg + loss[i] / (i + 1)
        avg_regret.append(cur_avg)
    return avg_regret

def test_simple_boost_lstm(steps=500, show=True):
    # model initialize
    T = steps
    model_id = "LSTM"
    ogd = OGD(learning_rate=0.01)
    model_params = {'n':1, 'm':1, 'l':5, 'h':10, 'optimizer':ogd}
    models = []
    Ns = [1, 3, 6]
    for n in Ns: # number of weak learners
        model = ctsb.model("SimpleBoost")
        model.initialize(model_id, model_params, n, reg=1.0) # regularization
        models.append(model)

    # regular AutoRegressor for comparison
    autoreg = ctsb.model("AutoRegressor")
    autoreg.initialize(p=4) # regularization

    # problem initialize
    p, q = 4, 0
    problem = ctsb.problem("ARMA-v0")
    y_true = problem.initialize(p, q, noise_magnitude=0.1)
 
    # run all boosting model
    result_list = [[] for n in Ns]
    last_value = []
    autoreg_loss = []
    for i in range(T):
        y_next = problem.step()

        # predictions for every boosting model
        for result_i, model_i in zip(result_list, models):
            y_pred = model_i.predict(y_true)
            result_i.append(mse(y_next, y_pred))
            model_i.update(y_next)

        # last value and autoregressor predictions
        last_value.append(mse(y_true, y_next))
        autoreg_loss.append(mse(autoreg.predict(y_true), y_next))
        autoreg.update(y_next)
        y_true = y_next
            
    # plot performance
    if show:
        start = 100
        x = np.arange(start, steps)
        plt.figure(figsize=(12,8))

        # plot every boosting model loss
        for n, results in zip(Ns, result_list):
            print("Mean loss for n={}: {}".format(n, np.mean(np.array(results[start:]))))
            plt.plot(x, avg_regret(results[start:]), label="SimpleBoost, n={}".format(n))

        # plot loss for last value and autoregressor models
        print("Mean loss for LastValue: {}".format(np.mean(np.array(last_value[start:]))))
        plt.plot(x, avg_regret(last_value[start:]), label="Last value model")
        print("Mean loss for AutoRegressor: {}".format(np.mean(np.array(autoreg_loss[start:]))))
        plt.plot(x, avg_regret(autoreg_loss[start:]), label="AutoRegressor model")

        plt.title("SimpleBoost model on ARMA problem")
        plt.legend()
        plt.show(block=False)
        plt.pause(10)
        plt.close()


def test_simple_boost_arma(steps=500, show=True):
    # model initialize
    T = steps
    model_id = "AutoRegressor"
    ogd = OGD(learning_rate=0.01)
    model_params = {'p':4, 'optimizer':ogd}
    models = []
    Ns = [1, 2, 3, 5, 8]
    for n in Ns: # number of weak learners
        model = ctsb.model("SimpleBoost")
        model.initialize(model_id, model_params, n, reg=1.0) # regularization
        models.append(model)

    # regular AutoRegressor for comparison
    autoreg = ctsb.model("AutoRegressor")
    autoreg.initialize(p=4) # regularization

    # problem initialize
    p, q = 4,1
    problem = ctsb.problem("ARMA-v0")
    y_true = problem.initialize(p,q)
 
    # run all boosting model
    result_list = [[] for n in Ns]
    last_value = []
    autoreg_loss = []
    for i in range(T):
        y_next = problem.step()

        # predictions for every boosting model
        for result_i, model_i in zip(result_list, models):
            y_pred = model_i.predict(y_true)
            result_i.append(mse(y_next, y_pred))
            model_i.update(y_next)

        # last value and autoregressor predictions
        last_value.append(mse(y_true, y_next))
        autoreg_loss.append(mse(autoreg.predict(y_true), y_next))
        autoreg.update(y_next)
        y_true = y_next
            
    # plot performance
    if show:
        start = 100
        x = np.arange(start, steps)
        plt.figure(figsize=(12,8))

        # plot every boosting model loss
        for n, results in zip(Ns, result_list):
            print("Mean loss for n={}: {}".format(n, np.mean(np.array(results[start:]))))
            plt.plot(x, avg_regret(results[start:]), label="SimpleBoost, n={}".format(n))

        # plot loss for last value and autoregressor models
        print("Mean loss for LastValue: {}".format(np.mean(np.array(last_value[start:]))))
        plt.plot(x, avg_regret(last_value[start:]), label="Last value model")
        print("Mean loss for AutoRegressor: {}".format(np.mean(np.array(autoreg_loss[start:]))))
        plt.plot(x, avg_regret(autoreg_loss[start:]), label="AutoRegressor model")

        plt.title("SimpleBoost model on ARMA problem")
        plt.legend()
        plt.show(block=False)
        plt.pause(10)
        plt.close()



if __name__ == "__main__":
    test_simple_boost(show=True)