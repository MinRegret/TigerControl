
import ctsb
from ctsb.models.optimizers import *
from ctsb.models.optimizers.losses import mse
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

    #ctsb.set_key(0) # consistent randomness

    problem = ctsb.problem('ARMA-v0')
    x, y_true = problem.initialize()

    models = []
    labels = ['OGD', 'ONS', 'Adam'] # don't run deprecated ONS

    model = ctsb.model('LSTM')
    model.initialize(n = 1, m = 1, optimizer=OGD) # initialize with class
    models.append(model)

    #model = ctsb.model('AutoRegressor')
    #model.initialize(optimizer=Adagrad) # initialize with class
    #models.append(model)

    model = ctsb.model('LSTM')
    model.initialize(n = 1, m = 1, optimizer=ONS) # initialize with class
    models.append(model)

    #model = ctsb.model('AutoRegressor')
    #model.initialize(optimizer=Adam) # initialize with class
    #models.append(model)

    losses = [[] for i in range(len(models))]
    update_time = [0.0 for i in range(len(models))]
    for t in tqdm(range(2000)):
        for i in range(len(models)):
            l, model = losses[i], models[i]
            y_pred = model.predict(x)
            l.append(mse(y_pred, y_true))

            t = time.time()
            model.update(y_true)
            update_time[i] += time.time() - t
        x, y_true = problem.step()

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