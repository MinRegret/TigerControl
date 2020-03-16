
import tigercontrol
from tigercontrol.utils.optimizers import *
from tigercontrol.utils.optimizers.losses import mse
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def avg_regret(loss):
        avg_regret = []
        cur_avg = 0
        for i in range(len(loss)):
            cur_avg = (i / (i + 1)) * cur_avg + loss[i] / (i + 1)
            avg_regret.append(cur_avg)
        return avg_regret

def test_ons(show=False):

    #tigercontrol.set_key(0) # consistent randomness

    environment = tigercontrol.environment('LDS')
    x, y_true = environment.reset()

    controllers = []
    labels = ['OGD', 'ONS', 'Adam'] # don't run deprecated ONS

    controller = tigercontrol.controllers('LSTM')
    controller.initialize(n = 1, m = 1, optimizer=OGD) # initialize with class
    controllers.append(controller)

    #controller = tigercontrol.controllers('AutoRegressor')
    #controller.initialize(optimizer=Adagrad) # initialize with class
    #controllers.append(controller)

    controller = tigercontrol.controllers('LSTM')
    controller.initialize(n = 1, m = 1, optimizer=ONS) # initialize with class
    controllers.append(controller)

    #controller = tigercontrol.controllers('AutoRegressor')
    #controller.initialize(optimizer=Adam) # initialize with class
    #controllers.append(controller)

    losses = [[] for i in range(len(controllers))]
    update_time = [0.0 for i in range(len(controllers))]
    for t in tqdm(range(2000)):
        for i in range(len(controllers)):
            l, controller = losses[i], controllers[i]
            y_pred = controller.predict(x)
            l.append(mse(y_pred, y_true))

            t = time.time()
            controller.update(y_true)
            update_time[i] += time.time() - t
        x, y_true = environment.step()

    print("time taken:")
    for t, label in zip(update_time, labels):
        print(label + ": " + str(t))

    if show:
        plt.yscale('log')
        for l, label in zip(losses, labels):
            plt.plot(l, label = label)
            #plt.plot(avg_regret(l), label = label)
        plt.legend()
        plt.title("Autoregressors on ENSO-T6")
        plt.show(block=False)
        plt.pause(300)
        plt.close()
        
    print("test_ons passed")

if __name__ == "__main__":
    test_ons(show=True)