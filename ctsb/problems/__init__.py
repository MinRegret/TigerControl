# problems init file

from ctsb.problems.registration import registry, register, make, spec


register(
    id='Random-v0',
    entry_point='ctsb.problems.time_series:Random',
    max_episode_steps=1000,
)