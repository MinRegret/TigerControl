import tigercontrol
from tigercontrol.experiments import Experiment
from tigercontrol.utils.download_tools import get_tigercontrol_dir
from tigercontrol.models.optimizers import *
import os

def test_new_experiment(steps=5000, show=False):
    exp = Experiment()
    exp.initialize(problems = ['SP500-v0'], models =  ['LastValue'], \
                    use_precomputed = False, timesteps = steps, verbose = show, load_bar = show)

    '''exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : OGD}, name = 'OGD')
                exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : Adagrad})
                exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : ONS})
                exp.add_model('AutoRegressor', {'p' : 3, 'optimizer' : Adam})
            
                exp.add_model('SimpleBoost', {'model_id': 'AutoRegressor', \
                    'model_params': {'p' : 3, 'optimizer' : OGD}}, name = 'AR-OGD')
                exp.add_model('SimpleBoost', {'model_id': 'AutoRegressor', \
                    'model_params': {'p' : 3, 'optimizer' : Adagrad}}, name = 'AR-Adagrad')
                exp.add_model('SimpleBoost', {'model_id': 'AutoRegressor', \
                    'model_params': {'p' : 3, 'optimizer' : ONS}}, name = 'AR-ONS')
                exp.add_model('SimpleBoost', {'model_id': 'AutoRegressor', \
                    'model_params': {'p' : 3, 'optimizer' : Adam}}, name = 'AR-Adam')'''

    '''exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : OGD}, name = 'OGD')
                exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : Adagrad})
                #exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : ONS})
                exp.add_model('RNN', {'n' : 1, 'm' : 1, 'optimizer' : Adam})
            
                exp.add_model('SimpleBoost', {'model_id': 'RNN', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : OGD}}, name = 'RNN-OGD')
                exp.add_model('SimpleBoost', {'model_id': 'RNN', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : Adagrad}}, name = 'RNN-Adagrad')
                exp.add_model('SimpleBoost', {'model_id': 'RNN', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : ONS}}, name = 'RNN-ONS')
                exp.add_model('SimpleBoost', {'model_id': 'RNN', \
                    'model_params': {'n' : 1, 'm' : 1, 'optimizer' : Adam}}, name = 'RNN-Adam')'''

    exp.add_model('LSTM', {'n' : 1, 'm' : 1, 'l' : 32, 'h' : 64, 'optimizer' : OGD}, name = '32-64-OGD')
    exp.add_model('LSTM', {'n' : 1, 'm' : 1, 'l' : 32, 'h' : 64, 'optimizer' : Adagrad}, name = '32-64-Adagrad')
    exp.add_model('LSTM', {'n' : 1, 'm' : 1, 'l' : 32, 'h' : 64,  'optimizer' : Adam}, name = '32-64-Adam')

    exp.add_model('SimpleBoost', {'model_id': 'LSTM', \
        'model_params': {'n' : 1, 'm' : 1, 'optimizer' : OGD}}, name = 'LSTM-OGD')
    exp.add_model('SimpleBoost', {'model_id': 'LSTM', \
        'model_params': {'n' : 1, 'm' : 1, 'optimizer' : Adagrad}}, name = 'LSTM-Adagrad')
    exp.add_model('SimpleBoost', {'model_id': 'LSTM', \
        'model_params': {'n' : 1, 'm' : 1, 'optimizer' : Adam}}, name = 'LSTM-Adam')

    exp.add_problem('ARMA-v0', {'p':2, 'q':0}, name = '20')
    #exp.add_problem('ARMA-v0', {'p':3, 'q':3}, name = '33')
    #exp.add_problem('ARMA-v0', {'p':5, 'q':4}, name = '54')

    #exp.add_problem('LDS-TimeSeries-v0', {'n': 1, 'm': 1, 'd':3}, name = '3')
    #exp.add_problem('LDS-TimeSeries-v0', {'n': 1, 'm': 1, 'd':10}, name = '10')

    #exp.add_problem('LSTM-TimeSeries-v0', {'n': 1, 'm': 1, 'h':32}, name = '32')
    #exp.add_problem('LSTM-TimeSeries-v0', {'n': 1, 'm': 1, 'h':64}, name = '64')

    exp.add_problem('ENSO-v0', {'input_signals': ['oni']}, name = 'T1')
    exp.add_problem('ENSO-v0', {'input_signals': ['oni'], 'timeline' : 3}, name = 'T3')
    exp.add_problem('ENSO-v0', {'input_signals': ['oni'], 'timeline' : 6}, name = 'T6')
    exp.add_problem('ENSO-v0', {'input_signals': ['oni'], 'timeline' : 12}, name = 'T12')

    tigercontrol_dir = get_tigercontrol_dir()
    datapath = 'data/results/scoreboard_sim_LSTM.csv'
    datapath = os.path.join(tigercontrol_dir, datapath)

    exp.scoreboard(save_as = datapath)
    #exp.scoreboard(metric = 'time')

    datapath = 'data/results/graph_sim_LSTM.png'
    datapath = os.path.join(tigercontrol_dir, datapath)

    exp.graph(yscale = 'log', save_as = datapath, dpi = 300)

    '''datapath = 'data/results/graph_sim_autoregressors.png'
                datapath = os.path.join(tigercontrol_dir, datapath)
            
                exp.graph(problem_ids = ['SP500-v0', 'ENSO-T1', 'ENSO-T3', 'ENSO-T6', 'ENSO-T12'], \
                    yscale = 'log', save_as = datapath, dpi = 200)'''

    print("test_new_experiment passed")

if __name__ == "__main__":
    test_new_experiment(show=True)
    
