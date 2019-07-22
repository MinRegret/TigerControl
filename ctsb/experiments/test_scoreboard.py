import ctsb
from ctsb.experiments.experiment import Experiment

def test_scoreboard(steps=100, show=False):
	exp = Experiment()
	exp.initialize(problems = ['Random-v0', 'ARMA-v0', 'SP500-v0', 'Crypto-v0'], models =  ['LastValue', 'AutoRegressor', 'PredictZero', 'ArmaOgd'])
	exp.scoreboard()

if __name__ == "__main__":
    test_scoreboard()