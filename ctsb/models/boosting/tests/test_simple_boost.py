""" 
test SimpleBoost model
note: n=3 seems to do consistently better than other n's...
"""
import jax.numpy as np
import ctsb
from ctsb.models.optimizers.ogd import OGD
from ctsb.models.optimizers.losses import mse
import matplotlib.pyplot as plt
from ctsb.utils.random import set_key
from tqdm import tqdm

def avg_regret(loss):
        avg_regret = []
        cur_avg = 0
        for i in range(len(loss)):
            cur_avg = (i / (i + 1)) * cur_avg + loss[i] / (i + 1)
            avg_regret.append(cur_avg)
        return avg_regret

def test_simple_boost(steps=5000, show=False):
    test_simple_boost_arma(steps=steps, show=show)
    #test_simple_boost_lstm(steps=steps, show=show)
    print("test_simple_boost passed")


def test_simple_boost_lstm(steps=500, show=True):
    # model initialize
    T = steps
    model_id = "LSTM"
    model_params = {'n':1, 'm':1, 'l':5, 'h':10, 'optimizer':Adagrad}
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
            plt.plot(x, results[start:], label="SimpleBoost, n={}".format(n))

        # plot loss for last value and autoregressor models
        print("Mean loss for LastValue: {}".format(np.mean(np.array(last_value[start:]))))
        plt.plot(x, last_value[start:], label="Last value model")
        print("Mean loss for AutoRegressor: {}".format(np.mean(np.array(autoreg_loss[start:]))))
        plt.plot(x, autoreg_loss[start:], label="AutoRegressor model")

        plt.title("SimpleBoost model on ARMA problem")
        plt.legend()
        plt.show(block=False)
        plt.pause(10)
        plt.close()


def test_simple_boost_arma(steps=500, show=True):
    # model initialize
    T = steps
    model_id = "AutoRegressor"
    model_params = {'p':18, 'optimizer':OGD}
    Ns = [64]
    timelines = [6, 9, 12]

    # regular AutoRegressor for comparison
    autoreg = ctsb.model("AutoRegressor")
    autoreg.initialize(p=18, optimizer = OGD) 

    fig, ax = plt.subplots(nrows=1, ncols=3)
    cur = 0

    # run all boosting model
    for timeline in timelines:

        # problem initialize
        problem = ctsb.problem("ENSO-v0")
        x, y_true = problem.initialize(input_signals = ['oni'], timeline = timeline)
        models = []

        for n in Ns: # number of weak learners
            model = ctsb.model("SimpleBoost")
            model.initialize(model_id, model_params, n, reg=0.0) # regularization
            models.append(model)

        result_list = [[] for n in Ns]
        autoreg_loss = []

        for i in tqdm(range(T)):

            # predictions for every boosting model
            for result_i, model_i in zip(result_list, models):
                y_pred = model_i.predict(x)
                result_i.append(mse(y_true, y_pred))
                model_i.update(y_true)

            # last value and autoregressor predictions
            autoreg_loss.append(mse(autoreg.predict(x), y_true))
            autoreg.update(y_true)
            x, y_true = problem.step()
            
        # plot performance
        if show:

            start = T//2

            # plot every boosting model loss
            for n, results in zip(Ns, result_list):
                print("Mean loss for n={}: {}".format(n, np.mean(np.array(results))))
                ax[cur].plot(avg_regret(results[-start:]), label="SimpleBoost, n={}".format(n))

            # plot loss for last value and autoregressor models
            print("Mean loss for AutoRegressor: {}".format(np.mean(np.array(autoreg_loss))))
            ax[cur].plot(avg_regret(autoreg_loss[-start:]), label="AutoRegressor model")
            ax[cur].legend(loc="upper right", fontsize=8)

        cur += 1

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_simple_boost(show=True)