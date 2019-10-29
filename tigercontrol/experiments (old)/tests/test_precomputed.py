import tigercontrol
from tigercontrol.experiments.experiment import Experiment
from tigercontrol.methods.optimizers import *

def test_precomputed(steps=100, show=False):
    exp = Experiment()
    exp.initialize(problems = ['Random-v0', 'ARMA-v0', 'SP500-v0', 'Crypto-v0', 'ENSO-v0'], \
    	methods =  ['PredictZero', 'LastValue'], verbose = show, load_bar = show)
    exp.add_method('AutoRegressor', {'p' : 2, 'optimizer' : Adagrad})
    #exp.scoreboard() # these two lines throw errors, please fix!!!
    #exp.graph()
    print("test_precomputed passed")

if __name__ == "__main__":
    test_precomputed()