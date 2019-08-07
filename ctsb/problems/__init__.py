# problems init file

from ctsb.problems.core import Problem
from ctsb.problems.registration import problem_registry, problem_register, problem
from ctsb.problems.time_series import TimeSeriesProblem
from ctsb.problems.control import ControlProblem
from ctsb.problems.custom import register_custom_problem, CustomProblem


# ---------- Control ----------

problem_register(
    id='CartPole-v0',
    entry_point='ctsb.problems.control:CartPole',
)

problem_register(
    id='Pendulum-v0',
    entry_point='ctsb.problems.control:Pendulum',
)

problem_register(
    id='DoublePendulum-v0',
    entry_point='ctsb.problems.control:DoublePendulum',
)


problem_register(
    id='LDS-Control-v0',
    entry_point='ctsb.problems.control:LDS_Control',
)

problem_register(
    id='RNN-Control-v0',
    entry_point='ctsb.problems.control:RNN_Control',
)

problem_register(
    id='LSTM-Control-v0',
    entry_point='ctsb.problems.control:LSTM_Control',
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
    id='UCI-Indoor-v0',
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

problem_register(
    id='LDS-TimeSeries-v0',
    entry_point='ctsb.problems.time_series:LDS_TimeSeries',
)

problem_register(
    id='RNN-TimeSeries-v0',
    entry_point='ctsb.problems.time_series:RNN_TimeSeries',
)

problem_register(
    id='LSTM-TimeSeries-v0',
    entry_point='ctsb.problems.time_series:LSTM_TimeSeries',
)


# ---------- PyBullet ----------


problem_register(
    id='PyBullet-Obstacles-v0',
    entry_point='ctsb.problems.pybullet:Obstacles'
)

problem_register(
    id='PyBullet-CartPole-v0',
    entry_point='ctsb.problems.pybullet:CartPole'
)

problem_register(
    id='PyBullet-CartPoleSwingup-v0',
    entry_point='ctsb.problems.pybullet:CartPoleSwingup'
)

problem_register(
    id='PyBullet-CartPoleDouble-v0',
    entry_point='ctsb.problems.pybullet:CartPoleDouble'
)

problem_register(
    id='PyBullet-Kuka-v0',
    entry_point='ctsb.problems.pybullet:Kuka'
)

problem_register(
    id='PyBullet-KukaDiverse-v0',
    entry_point='ctsb.problems.pybullet:KukaDiverse'
)

problem_register(
    id='PyBullet-Minitaur-v0',
    entry_point='ctsb.problems.pybullet:Minitaur'
)

problem_register(
    id='PyBullet-HalfCheetah-v0',
    entry_point='ctsb.problems.pybullet:HalfCheetah'
)

problem_register(
    id='PyBullet-Ant-v0',
    entry_point='ctsb.problems.pybullet:Ant'
)

problem_register(
    id='PyBullet-Humanoid-v0',
    entry_point='ctsb.problems.pybullet:Humanoid'
)



