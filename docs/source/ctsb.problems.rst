tigercontrol.problems package
=====================

.. automodule:: tigercontrol.problems

core
----

This is a core

.. autosummary::
  :toctree: _autosummary

   Problem

custom
------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.problems.CustomProblem
   tigercontrol.problems.register_custom_problem


control
-------

.. toctree::
   :maxdepth: 3

.. autosummary::
  :toctree: _autosummary

   tigercontrol.problems.ControlProblem
   tigercontrol.problems.control.LDS_Control
   tigercontrol.problems.control.LSTM_Control
   tigercontrol.problems.control.RNN_Control
   tigercontrol.problems.control.CartPole
   tigercontrol.problems.control.DoublePendulum
   tigercontrol.problems.control.Pendulum


time_series
-----------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.problems.TimeSeriesProblem
   tigercontrol.problems.time_series.SP500
   tigercontrol.problems.time_series.UCI_Indoor
   tigercontrol.problems.time_series.ENSO
   tigercontrol.problems.time_series.Crypto
   tigercontrol.problems.time_series.Random
   tigercontrol.problems.time_series.ARMA
   tigercontrol.problems.time_series.Unemployment
   tigercontrol.problems.time_series.LDS_TimeSeries
   tigercontrol.problems.time_series.LSTM_TimeSeries
   tigercontrol.problems.time_series.RNN_TimeSeries

pybullet
--------

.. autosummary::
  :toctree: _autosummary

  tigercontrol.problems.pybullet.PyBulletProblem
  tigercontrol.problems.pybullet.Simulator
  tigercontrol.problems.pybullet.Ant
  tigercontrol.problems.pybullet.CartPole
  tigercontrol.problems.pybullet.CartPoleDouble
  tigercontrol.problems.pybullet.CartPoleSwingup
  tigercontrol.problems.pybullet.HalfCheetah
  tigercontrol.problems.pybullet.Humanoid
  tigercontrol.problems.pybullet.Kuka
  tigercontrol.problems.pybullet.KukaDiverse
  tigercontrol.problems.pybullet.Minitaur
  tigercontrol.problems.pybullet.Obstacles
  

