
import ctsb
from ctsb.models.optimizers.ons import ONS
from ctsb.models.optimizers.deprecated_ons import deprecated_ONS
from ctsb.models.optimizers.losses import mse
import matplotlib.pyplot as plt
import time

def test_ons(show=False):

    #ctsb.set_key(0) # consistent randomness

    problem = ctsb.problem('ARMA-v0')
    x = problem.initialize(p=2,q=0)

    models = []
    #labels = ['ONS + project', 'ONS no project', 'Deprecated ONS']
    labels = ['ONS + project', 'ONS no project'] # don't run deprecated ONS

    model = ctsb.model('LSTM')
    model.initialize(n=1, m=1, l=3, h=10, optimizer=ONS) # initialize with class
    models.append(model)

    model = ctsb.model('LSTM')
    optimizer = ONS(hyperparameters={'project':False})
    model.initialize(n=1, m=1, l=3, h=10, optimizer=optimizer) # initialize with class
    models.append(model)

    #model = ctsb.model('LSTM')
    #model.initialize(n=1, m=1, l=3, h=10, optimizer=deprecated_ONS) # initialize with class
    #models.append(model)

    losses = [[] for i in range(len(models))]
    update_time = [0.0 for i in range(len(models))]
    for t in range(100):
        y_true = problem.step()
        for i in range(len(models)):
            l, model = losses[i], models[i]
            y_pred = model.predict(x)
            l.append(mse(y_pred, y_true))

            t = time.time()
            model.update(y_true)
            update_time[i] += time.time() - t
        x = y_true

    print("time taken:")
    for t, label in zip(update_time, labels):
        print(label + ": " + str(t))

    if show:
        plt.yscale('log')
        for l, label in zip(losses, labels):
            plt.plot(l, label = label)
        plt.legend()
        plt.show(block=False)
        plt.pause(30)
        plt.close()
        
    print("test_ons passed")

if __name__ == "__main__":
    test_ons(show=True)