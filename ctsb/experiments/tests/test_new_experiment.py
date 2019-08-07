import ctsb
from ctsb.experiments import Experiment
from ctsb.utils.download_tools import get_ctsb_dir
from ctsb.models.optimizers import *
import os

def test_new_experiment(steps=50, show=False):
    exp = Experiment()
    exp.initialize(problems = ['Crypto-v0'], \
        models =  ['LastValue'], use_precomputed = False, \
        timesteps = steps, verbose = show, load_bar = show)

    exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : OGD}, name = 'OGD')
    exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : Adagrad})
    exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : ONS})
    exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : Adam})

    exp.add_model('SimpleBoost', {'model_id': 'AutoRegressor', \
        'model_params': {'p' : 3, 'optimizer' : OGD}}, name = 'AR-OGD')
    exp.add_model('SimpleBoost', {'model_id': 'AutoRegressor', \
        'model_params': {'p' : 3, 'optimizer' : Adagrad}}, name = 'AR-Adagrad')
    #exp.add_model('SimpleBoost', {'model_id': 'AutoRegressor', \
    #    'model_params': {'p' : 3, 'optimizer' : ONS}}, name = 'AR-ONS')
    exp.add_model('SimpleBoost', {'model_id': 'AutoRegressor', \
        'model_params': {'p' : 3, 'optimizer' : Adam}}, name = 'AR-Adam')

    '''exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : OGD}, name = 'OGD')
                exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : Adagrad})
                exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : ONS})
                exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : Adam})
            
                exp.add_model('LSTM', {'n' : 1, 'm' : 1, 'optimizer' : OGD}, name = 'OGD')
                exp.add_model('LSTM', {'n' : 1, 'm' : 1, 'optimizer' : Adagrad})
                exp.add_model('LSTM', {'n' : 1, 'm' : 1, 'optimizer' : ONS})
                exp.add_model('LSTM', {'n' : 1, 'm' : 1, 'optimizer' : Adam})'''

    '''exp.add_model('SimpleBoost', {'model_id': 'RNN', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : OGD}}, name = 'RNN-OGD')
                exp.add_model('SimpleBoost', {'model_id': 'RNN', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : Adagrad}}, name = 'RNN-Adagrad')
                exp.add_model('SimpleBoost', {'model_id': 'RNN', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : ONS}}, name = 'RNN-ONS')
                exp.add_model('SimpleBoost', {'model_id': 'RNN', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : Adam}}, name = 'RNN-Adam')
            
                exp.add_model('SimpleBoost', {'model_id': 'LSTM', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : OGD}}, name = 'LSTM-OGD')
                exp.add_model('SimpleBoost', {'model_id': 'LSTM', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : Adagrad}}, name = 'LSTM-Adagrad')
                exp.add_model('SimpleBoost', {'model_id': 'LSTM', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : ONS}}, name = 'LSTM-ONS')
                exp.add_model('SimpleBoost', {'model_id': 'LSTM', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : Adam}}, name = 'LSTM-Adam')'''

    exp.add_problem('ARMA-v0', {'p':2, 'q':0}, name = '20')
    exp.add_problem('ARMA-v0', {'p':3, 'q':3}, name = '33')
    exp.add_problem('ARMA-v0', {'p':5, 'q':4}, name = '54')

    #exp.add_problem('ENSO-v0', {'input_signals': ['oni']}, name = '1-month')
    exp.add_problem('ENSO-v0', {'input_signals': ['oni'], 'timeline' : 3}, name = '3-months')
    exp.add_problem('ENSO-v0', {'input_signals': ['oni'], 'timeline' : 6}, name = '6-months')
    #exp.add_problem('ENSO-v0', {'input_signals': ['oni'], 'timeline' : 12}, name = '12-months')

    ctsb_dir = get_ctsb_dir()
    datapath = 'data/results/scoreboard.csv'
    datapath = os.path.join(ctsb_dir, datapath)

    exp.scoreboard2(save_as = datapath)
    #exp.scoreboard(metric = 'time')

    datapath = 'data/results/graph.png'
    datapath = os.path.join(ctsb_dir, datapath)

    #exp.graph(save_as = datapath)
    exp.graph(yscale = 'log')
    print("test_new_experiment passed")

if __name__ == "__main__":
    test_new_experiment()