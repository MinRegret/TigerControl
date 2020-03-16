""" 
test DynaBoost controller
note: n=3 seems to do consistently better than other n's...
"""
import jax.numpy as np
import tigercontrol
from tigercontrol.utils.optimizers.ogd import OGD
from tigercontrol.utils.optimizers.losses import mse
import matplotlib.pyplot as plt
from tigercontrol.utils.random import set_key
from tqdm import tqdm

def avg_regret(loss):
        avg_regret = []
        cur_avg = 0
        for i in range(len(loss)):
            cur_avg = (i / (i + 1)) * cur_avg + loss[i] / (i + 1)
            avg_regret.append(cur_avg)
        return avg_regret

def test_dynaboost(steps=5000, show=False):
    test_dynaboost_arma(steps=steps, show=show)
    #test_dynaboost_lstm(steps=steps, show=show)
    print("test_dynaboost passed")

def avg_regret(loss):
    avg_regret = []
    cur_avg = 0
    for i in range(len(loss)):
        cur_avg = (i / (i + 1)) * cur_avg + loss[i] / (i + 1)
        avg_regret.append(cur_avg)
    return avg_regret

def test_dynaboost_lstm(steps=500, show=True):
    # controller initialize
    T = steps
    controller_id = "LSTM"
    ogd = OGD(learning_rate=0.01)
    controller_params = {'n':1, 'm':1, 'l':5, 'h':10, 'optimizer':ogd}
    controllers = []
    Ns = [1, 3, 6]
    for n in Ns: # number of weak learners
        controller = tigercontrol.controllers("DynaBoost")
        controller.initialize(controller_id, controller_params, n, reg=1.0) # regularization
        controllers.append(controller)

    # regular AutoRegressor for comparison
    autoreg = tigercontrol.controllers("AutoRegressor")
    autoreg.initialize(p=4) # regularization

    # environment initialize
    p, q = 4, 0
    environment = tigercontrol.environment("LDS")
    y_true = environment.reset(p, q, noise_magnitude=0.1)
 
    # run all boosting controller
    result_list = [[] for n in Ns]
    last_value = []
    autoreg_loss = []
    for i in range(T):
        y_next = environment.step()

        # predictions for every boosting controller
        for result_i, controller_i in zip(result_list, controllers):
            y_pred = controller_i.predict(y_true)
            result_i.append(mse(y_next, y_pred))
            controller_i.update(y_next)

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

        # plot every boosting controller loss
        for n, results in zip(Ns, result_list):
            print("Mean loss for n={}: {}".format(n, np.mean(np.array(results[start:]))))
            plt.plot(x, avg_regret(results[start:]), label="DynaBoost, n={}".format(n))

        # plot loss for last value and autoregressor controllers
        print("Mean loss for LastValue: {}".format(np.mean(np.array(last_value[start:]))))
        plt.plot(x, avg_regret(last_value[start:]), label="Last value controller")
        print("Mean loss for AutoRegressor: {}".format(np.mean(np.array(autoreg_loss[start:]))))
        plt.plot(x, avg_regret(autoreg_loss[start:]), label="AutoRegressor controller")

        plt.title("DynaBoost controller on LQR environment")
        plt.legend()
        plt.show(block=False)
        plt.pause(10)
        plt.close()


def test_dynaboost_arma(steps=500, show=True):
    # controller initialize
    T = steps
    controller_id = "AutoRegressor"
    controller_params = {'p':18, 'optimizer':OGD}
    Ns = [64]
    timelines = [6, 9, 12]

    # regular AutoRegressor for comparison
    autoreg = tigercontrol.controllers("AutoRegressor")
    autoreg.initialize(p=18, optimizer = OGD) 

    fig, ax = plt.subplots(nrows=1, ncols=3)
    cur = 0

    # run all boosting controller
    for timeline in timelines:

        # environment initialize
        environment = tigercontrol.environment("ENSO")
        x, y_true = environment.reset(input_signals = ['oni'], timeline = timeline)
        controllers = []

        for n in Ns: # number of weak learners
            controller = tigercontrol.controllers("DynaBoost")
            controller.initialize(controller_id, controller_params, n, reg=0.0) # regularization
            controllers.append(controller)

        result_list = [[] for n in Ns]
        autoreg_loss = []

        for i in tqdm(range(T)):

            # predictions for every boosting controller
            for result_i, controller_i in zip(result_list, controllers):
                y_pred = controller_i.predict(x)
                result_i.append(mse(y_true, y_pred))
                controller_i.update(y_true)

            # last value and autoregressor predictions
            autoreg_loss.append(mse(autoreg.predict(x), y_true))
            autoreg.update(y_true)
            x, y_true = environment.step()
            
        # plot performance
        if show:

            start = T//2

            # plot every boosting controller loss
            for n, results in zip(Ns, result_list):
                print("Mean loss for n={}: {}".format(n, np.mean(np.array(results))))
                ax[cur].plot(avg_regret(results[-start:]), label="DynaBoost, n={}".format(n))

            # plot loss for last value and autoregressor controllers
            print("Mean loss for AutoRegressor: {}".format(np.mean(np.array(autoreg_loss))))
            ax[cur].plot(avg_regret(autoreg_loss[-start:]), label="AutoRegressor controller")
            ax[cur].legend(loc="upper right", fontsize=8)

        cur += 1

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_dynaboost(show=True)