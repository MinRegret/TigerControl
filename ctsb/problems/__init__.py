# problems init file

from tigercontrol.problems.core import Problem
from tigercontrol.problems.registration import problem_registry, problem_register, problem
from tigercontrol.problems.time_series import TimeSeriesProblem
from tigercontrol.problems.control import ControlProblem
from tigercontrol.problems.custom import register_custom_problem, CustomProblem


# ---------- Control ----------

problem_register(
    id='CartPole-v0',
    entry_point='tigercontrol.problems.control:CartPole',
)

problem_register(
    id='Pendulum-v0',
    entry_point='tigercontrol.problems.control:Pendulum',
)

problem_register(
    id='DoublePendulum-v0',
    entry_point='tigercontrol.problems.control:DoublePendulum',
)


problem_register(
    id='LDS-Control-v0',
    entry_point='tigercontrol.problems.control:LDS_Control',
)

problem_register(
    id='RNN-Control-v0',
    entry_point='tigercontrol.problems.control:RNN_Control',
)

problem_register(
    id='LSTM-Control-v0',
    entry_point='tigercontrol.problems.control:LSTM_Control',
)


# ---------- Time-series ----------


problem_register(
    id='Random-v0',
    entry_point='tigercontrol.problems.time_series:Random',
)

problem_register(
    id='ARMA-v0',
    entry_point='tigercontrol.problems.time_series:ARMA',
)

problem_register(
    id='SP500-v0',
    entry_point='tigercontrol.problems.time_series:SP500',
)

problem_register(
    id='UCI-Indoor-v0',
    entry_point='tigercontrol.problems.time_series:UCI_Indoor',
)

problem_register(
    id='Crypto-v0',
    entry_point='tigercontrol.problems.time_series:Crypto',
)

problem_register(
    id='Unemployment-v0',
    entry_point='tigercontrol.problems.time_series:Unemployment',
)

problem_register(
    id='ENSO-v0',
    entry_point='tigercontrol.problems.time_series:ENSO',
)

problem_register(
    id='LDS-TimeSeries-v0',
    entry_point='tigercontrol.problems.time_series:LDS_TimeSeries',
)

problem_register(
    id='RNN-TimeSeries-v0',
    entry_point='tigercontrol.problems.time_series:RNN_TimeSeries',
)

problem_register(
    id='LSTM-TimeSeries-v0',
    entry_point='tigercontrol.problems.time_series:LSTM_TimeSeries',
)


# ---------- PyBullet ----------


problem_register(
    id='PyBullet-Obstacles-v0',
    entry_point='tigercontrol.problems.pybullet:Obstacles'
)

problem_register(
    id='PyBullet-CartPole-v0',
    entry_point='tigercontrol.problems.pybullet:CartPole'
)

problem_register(
    id='PyBullet-CartPoleSwingup-v0',
    entry_point='tigercontrol.problems.pybullet:CartPoleSwingup'
)

problem_register(
    id='PyBullet-CartPoleDouble-v0',
    entry_point='tigercontrol.problems.pybullet:CartPoleDouble'
)

problem_register(
    id='PyBullet-Kuka-v0',
    entry_point='tigercontrol.problems.pybullet:Kuka'
)

problem_register(
    id='PyBullet-KukaDiverse-v0',
    entry_point='tigercontrol.problems.pybullet:KukaDiverse'
)

problem_register(
    id='PyBullet-Minitaur-v0',
    entry_point='tigercontrol.problems.pybullet:Minitaur'
)

problem_register(
    id='PyBullet-HalfCheetah-v0',
    entry_point='tigercontrol.problems.pybullet:HalfCheetah'
)

problem_register(
    id='PyBullet-Ant-v0',
    entry_point='tigercontrol.problems.pybullet:Ant'
)

problem_register(
    id='PyBullet-Humanoid-v0',
    entry_point='tigercontrol.problems.pybullet:Humanoid'
)



