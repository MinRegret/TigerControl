import ctsb
from ctsb.experiments import Experiment
from ctsb.models.optimizers import *

def test_new_experiment(steps=100, show=False):
    exp = Experiment()
    exp.initialize(problems = ['Random-v0', 'ARMA-v0', 'Crypto-v0', 'Unemployment-v0'], \
        models =  ['PredictZero', 'LastValue'], use_precomputed = False, \
        timesteps = steps, verbose = show, load_bar = show)
    exp.add_model('AutoRegressor', {'p' : 2, 'optimizer' : OGD}, name = 'OGD')
    exp.add_model('AutoRegressor', {'p' : 2, 'optimizer' : Adagrad})
    exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : OGD}, name = 'OGD')
    #exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : Adagrad})
    exp.add_model('LSTM', {'n' : 1, 'm' : 1, 'optimizer' : OGD}, name = 'OGD')
    #exp.add_model('LSTM', {'n' : 1, 'm' : 1, 'optimizer' : Adagrad})
    exp.scoreboard()
    exp.scoreboard(metric = 'time')
    exp.graph()
    print("test_new_experiment passed")

if __name__ == "__main__":
    test_new_experiment()