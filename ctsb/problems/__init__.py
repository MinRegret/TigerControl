# problems init file

from ctsb.problems.registration import problem_registry, problem_register, problem, problem_spec
from ctsb.problems.core import Problem
from ctsb.problems.time_series import TimeSeriesProblem
from ctsb.problems.control import ControlProblem


# ---------- Control ----------


problem_register(
    id='CartPole-v0',
    entry_point='ctsb.problems.control.pybullet:CartPole'
)

problem_register(
    id='CartPoleSwingup-v0',
    entry_point='ctsb.problems.control.pybullet:CartPoleSwingup'
)

problem_register(
    id='CartPoleDouble-v0',
    entry_point='ctsb.problems.control.pybullet:CartPoleDouble'
)


# ---------- Control ----------


problem_register(
    id='LDS-v0',
    entry_point='ctsb.problems.control:LDS',
)

problem_register(
    id='RNN-v0',
    entry_point='ctsb.problems.control:RNN_Output',
)

problem_register(
    id='LSTM-v0',
    entry_point='ctsb.problems.control:LSTM_Output',
)


# ---------- Time-series ----------


problem_register(
    id='Random-v0',
    entry_point='ctsb.problems.time_series:Random',
)

problem_register(
    id='ARMA-v0',
    entry_point='ctsb.problems.time_series:ARMA',
)

problem_register(
    id='SP500-v0',
    entry_point='ctsb.problems.time_series:SP500',
)

problem_register(
    id='UCIIndoor-v0',
    entry_point='ctsb.problems.time_series:UCI_Indoor',
)

problem_register(
    id='Crypto-v0',
    entry_point='ctsb.problems.time_series:Crypto',
)

problem_register(
    id='CtrlIndices-v0',
    entry_point='ctsb.problems.time_series:CtrlIndices',
)

