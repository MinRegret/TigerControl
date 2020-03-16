tigercontrol.environments package
=============================

.. automodule:: tigercontrol.environments

core
----

This is a core

.. autosummary::
  :toctree: _autosummary

   Environment

custom
------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.environments.CustomEnvironment
   tigercontrol.environments.register_custom_environment


control
-------

.. toctree::
   :maxdepth: 3

.. autosummary::
  :toctree: _autosummary

   tigercontrol.environments.ControlEnvironment
   tigercontrol.environments.controller.LDS_Control
   tigercontrol.environments.controller.LSTM_Control
   tigercontrol.environments.controller.RNN_Control
   tigercontrol.environments.controller.CartPole
   tigercontrol.environments.controller.DoublePendulum
   tigercontrol.environments.controller.Pendulum


time_series
-----------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.environments.TimeSeriesEnvironment
   tigercontrol.environments.time_series.SP500
   tigercontrol.environments.time_series.UCI_Indoor
   tigercontrol.environments.time_series.ENSO
   tigercontrol.environments.time_series.Crypto
   tigercontrol.environments.time_series.Random
   tigercontrol.environments.time_series.LQR
   tigercontrol.environments.time_series.Unemployment
   tigercontrol.environments.time_series.LDS_TimeSeries
   tigercontrol.environments.time_series.LSTM_TimeSeries
   tigercontrol.environments.time_series.RNN_TimeSeries

pybullet
--------

.. autosummary::
  :toctree: _autosummary

  tigercontrol.environments.pybullet.PyBulletEnvironment
  tigercontrol.environments.pybullet.Simulator
  tigercontrol.environments.pybullet.Ant
  tigercontrol.environments.pybullet.CartPole
  tigercontrol.environments.pybullet.CartPoleDouble
  tigercontrol.environments.pybullet.CartPoleSwingup
  tigercontrol.environments.pybullet.HalfCheetah
  tigercontrol.environments.pybullet.Humanoid
  tigercontrol.environments.pybullet.Kuka
  tigercontrol.environments.pybullet.KukaDiverse
  tigercontrol.environments.pybullet.Minitaur
  tigercontrol.environments.pybullet.Obstacles
  

