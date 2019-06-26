# problems init file

from ctsb.problems.registration import problem_registry, problem_register, problem, problem_spec
from ctsb.problems.core import Problem


# ---------- Simulated ----------


problem_register(
    id='Random-v0',
    entry_point='ctsb.problems.simulated:Random',
)

problem_register(
    id='ARMA-v0',
    entry_point='ctsb.problems.simulated:ARMA',
)

problem_register(
    id='LDS-v0',
    entry_point='ctsb.problems.simulated:LDS',
)

problem_register(
    id='RNN-v1',
    entry_point='ctsb.problems.simulated:RNN_Output',
)

problem_register(
    id='LSTM-v0',
    entry_point='ctsb.problems.simulated:LSTM_Output',
)


# ---------- Data based ----------


problem_register(
    id='SP500-v0',
    entry_point='ctsb.problems.data_based:SP500',
)

problem_register(
    id='UCI-Indoor-v0',
    entry_point='ctsb.problems.data_based:UCI_Indoor',
)

