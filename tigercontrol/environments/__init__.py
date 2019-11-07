# environments init file

from tigercontrol.environments.core import Environment
from tigercontrol.environments.registration import environment_registry, environment_register, environment
from tigercontrol.environments.custom import register_custom_environment, CustomEnvironment


# ---------- Control ----------

environment_register(
    id='CartPole-v0',
    entry_point='tigercontrol.environments.control:CartPole',
)

environment_register(
    id='Pendulum-v0',
    entry_point='tigercontrol.environments.control:Pendulum',
)

environment_register(
    id='Pendulum3State-v0',
    entry_point='tigercontrol.environments.control:Pendulum_3_State',
)

environment_register(
    id='DoublePendulum-v0',
    entry_point='tigercontrol.environments.control:DoublePendulum',
)

environment_register(
    id='LDS-v0',
    entry_point='tigercontrol.environments.control:LDS',
)

environment_register(
    id='LDS-Control-v0',
    entry_point='tigercontrol.environments.control:LDS_Control',
)

environment_register(
    id='RNN-Control-v0',
    entry_point='tigercontrol.environments.control:RNN_Control',
)

environment_register(
    id='LSTM-Control-v0',
    entry_point='tigercontrol.environments.control:LSTM_Control',
)

environment_register(
    id='PlanarQuadrotor-v0',
    entry_point='tigercontrol.environments.control:PlanarQuadrotor',
)

environment_register(
    id='Quadcopter-v0',
    entry_point='tigercontrol.environments.control:Quadcopter',
)

# ---------- PyBullet ----------


environment_register(
    id='PyBullet-Obstacles-v0',
    entry_point='tigercontrol.environments.pybullet:Obstacles'
)

environment_register(
    id='PyBullet-CartPole-v0',
    entry_point='tigercontrol.environments.pybullet:CartPole'
)

environment_register(
    id='PyBullet-CartPoleSwingup-v0',
    entry_point='tigercontrol.environments.pybullet:CartPoleSwingup'
)

environment_register(
    id='PyBullet-CartPoleDouble-v0',
    entry_point='tigercontrol.environments.pybullet:CartPoleDouble'
)

environment_register(
    id='PyBullet-Kuka-v0',
    entry_point='tigercontrol.environments.pybullet:Kuka'
)

environment_register(
    id='PyBullet-KukaDiverse-v0',
    entry_point='tigercontrol.environments.pybullet:KukaDiverse'
)

environment_register(
    id='PyBullet-Minitaur-v0',
    entry_point='tigercontrol.environments.pybullet:Minitaur'
)

environment_register(
    id='PyBullet-HalfCheetah-v0',
    entry_point='tigercontrol.environments.pybullet:HalfCheetah'
)

environment_register(
    id='PyBullet-Ant-v0',
    entry_point='tigercontrol.environments.pybullet:Ant'
)

environment_register(
    id='PyBullet-Humanoid-v0',
    entry_point='tigercontrol.environments.pybullet:Humanoid'
)
