import ctsb
from ctsb.experiments import Experiment
from ctsb.models.optimizers import *

def test_new_experiment(steps=100, show=False):
    exp = Experiment()
    exp.initialize(problems = ['Random-v0', 'ARMA-v0', 'SP500-v0', 'Crypto-v0', 'ENSO-v0'], \
        models =  ['PredictZero', 'LastValue', 'AutoRegressor'], use_precomputed = False, \
        timesteps = steps, verbose = show, load_bar = show)
    exp.add_model('AutoRegressor', {'p' : 2, 'optimizer' : Adagrad})
    exp.add_model('AutoRegressor', {'p' : 2, 'optimizer' : OGD})
    exp.scoreboard()
    exp.graph()
    print("test_new_experiment passed")

if __name__ == "__main__":
    test_new_experiment()