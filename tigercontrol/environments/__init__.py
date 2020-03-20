# environments and control init file

from tigercontrol.environments.core import Environment
from tigercontrol.environments.registration import environment_registry, environment_register, environment

from tigercontrol.environments.lds import LDS
from tigercontrol.environments.cartpole import CartPole, cartpole_basic_loss
from tigercontrol.environments.pendulum import Pendulum
from tigercontrol.environments.double_pendulum import DoublePendulum
from tigercontrol.environments.quadcopter import Quadcopter


# ---------- Control ----------

environment_register(
    id='LDS',
    entry_point='tigercontrol.environments:LDS',
)

environment_register(
    id='CartPole',
    entry_point='tigercontrol.environments:CartPole',
    kwargs={'loss': cartpole_basic_loss},
)

environment_register(
    id='Pendulum',
    entry_point='tigercontrol.environments:Pendulum',
)

environment_register(
    id='DoublePendulum',
    entry_point='tigercontrol.environments:DoublePendulum',
)

environment_register(
    id='Quadcopter',
    entry_point='tigercontrol.environments:Quadcopter',
)
