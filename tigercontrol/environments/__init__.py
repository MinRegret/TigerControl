# environments and control init file

from tigercontrol.environments.core import Environment
from tigercontrol.environments.registration import environment_registry, environment_register, environment

from tigercontrol.environments.lds import LDS
from tigercontrol.environments.lds_control import LDS_Control
from tigercontrol.environments.cartpole import CartPole
from tigercontrol.environments.pendulum import Pendulum
from tigercontrol.environments.double_pendulum import DoublePendulum
from tigercontrol.environments.quadcopter import Quadcopter


# ---------- Control ----------

environment_register(
    id='LDS-v0',
    entry_point='tigercontrol.environments:LDS',
)

environment_register(
    id='LDS-Control-v0',
    entry_point='tigercontrol.environments:LDS_Control',
)

environment_register(
    id='CartPole-v0',
    entry_point='tigercontrol.environments:CartPole',
)

environment_register(
    id='Pendulum-v0',
    entry_point='tigercontrol.environments:Pendulum',
)

environment_register(
    id='DoublePendulum-v0',
    entry_point='tigercontrol.environments:DoublePendulum',
)

environment_register(
    id='Quadcopter-v0',
    entry_point='tigercontrol.environments:Quadcopter',
)
