# problems init file

from ctsb.problems.core import Problem
from ctsb.problems.registration import problem_registry, problem_register, problem
from ctsb.problems.time_series import TimeSeriesProblem
from ctsb.problems.control import ControlProblem
from ctsb.problems.custom import register_custom_problem, CustomProblem


# ---------- PyBullet ----------


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

problem_register(
    id='Kuka-v0',
    entry_point='ctsb.problems.control.pybullet:Kuka'
)

problem_register(
    id='KukaDiverse-v0',
    entry_point='ctsb.problems.control.pybullet:KukaDiverse'
)

problem_register(
    id='Minitaur-v0',
    entry_point='ctsb.problems.control.pybullet:Minitaur'
)

problem_register(
    id='HalfCheetah-v0',
    entry_point='ctsb.problems.control.pybullet:HalfCheetah'
)

problem_register(
    id='Ant-v0',
    entry_point='ctsb.problems.control.pybullet:Ant'
)

problem_register(
    id='Humanoid-v0',
    entry_point='ctsb.problems.control.pybullet:Humanoid'
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
    id='Unemployment-v0',
    entry_point='ctsb.problems.time_series:Unemployment',
)

problem_register(
    id='ENSO-v0',
    entry_point='ctsb.problems.time_series:ENSO',
)

