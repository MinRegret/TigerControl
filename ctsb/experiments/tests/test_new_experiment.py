import ctsb
from ctsb.experiments import Experiment
from ctsb.models.optimizers import *

def test_new_experiment(steps=500, show=True):
    exp = Experiment()
    exp.initialize(problems = ['Random-v0', 'Crypto-v0', 'Unemployment-v0'], \
        models =  ['PredictZero', 'LastValue'], use_precomputed = False, \
        timesteps = steps, verbose = show, load_bar = show)

    exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : OGD}, name = 'OGD')
    exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : Adagrad})
    exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : ONS})
    exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : Adam})

    exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : OGD}, name = 'OGD')
    exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : Adagrad})

    exp.add_model('LSTM', {'n' : 1, 'm' : 1, 'optimizer' : OGD}, name = 'OGD')
    exp.add_model('LSTM', {'n' : 1, 'm' : 1, 'optimizer' : Adagrad})

    exp.add_problem('ARMA-v0', {'p':2, 'q':0}, name = '20')
    exp.add_problem('ARMA-v0', {'p':3, 'q':3}, name = '33')
    exp.add_problem('ARMA-v0', {'p':5, 'q':4}, name = '54')

    exp.add_problem('ENSO-v0', {'input_signals': ['oni']}, name = '1-month')
    exp.add_problem('ENSO-v0', {'input_signals': ['oni'], 'timeline' : 3}, name = '3-months')
    exp.add_problem('ENSO-v0', {'input_signals': ['oni'], 'timeline' : 6}, name = '6-months')
    exp.add_problem('ENSO-v0', {'input_signals': ['oni'], 'timeline' : 12}, name = '12-months')

    exp.scoreboard2()
    #exp.scoreboard(metric = 'time')
    exp.graph()
    print("test_new_experiment passed")

if __name__ == "__main__":
    test_new_experiment()