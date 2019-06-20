# problems init file

from ctsb.problems.registration import registry, register, problem, spec, help


# ---------- Simulated ----------


register(
    id='Random-v0',
    entry_point='ctsb.problems.simulated:Random',
    max_episode_steps=100000,
)

register(
    id='ARMA-v0',
    entry_point='ctsb.problems.simulated:ARMA',
    max_episode_steps=100000,
)

register(
    id='LDS-v0',
    entry_point='ctsb.problems.simulated:LDS',
    max_episode_steps=100000,
)

register(
    id='RNN-v0',
    entry_point='ctsb.problems.simulated:RNN_Output',
    max_episode_steps=100000,
)

register(
    id='LSTM-v0',
    entry_point='ctsb.problems.simulated:LSTM_Output',
    max_episode_steps=100000,
)


# ---------- Data based ----------


register(
    id='SP500-v0',
    entry_point='ctsb.problems.data_based:SP500',
    max_episode_steps=100000,
)
