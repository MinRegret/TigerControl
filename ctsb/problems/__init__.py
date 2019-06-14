# problems init file

from ctsb.problems.registration import registry, register, make, spec


# ----- Time series -----

register(
    id='Random-v0',
    entry_point='ctsb.problems.time_series:Random',
    max_episode_steps=100000,
)

register(
    id='ARMA-v0',
    entry_point='ctsb.problems.time_series:ARMA',
    max_episode_steps=100000,
)

register(
    id='LDS-v0',
    entry_point='ctsb.problems.time_series:LDS',
    max_episode_steps=100000,
)

register(
    id='RNN-v0',
    entry_point='ctsb.problems.time_series:RNN_Output',
    max_episode_steps=100000,
)

register(
    id='LSTM-v0',
    entry_point='ctsb.problems.time_series:LSTM_Output',
    max_episode_steps=100000,
)