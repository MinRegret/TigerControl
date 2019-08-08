ctsb.problems package
=====================

.. automodule:: ctsb.problems

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

   ctsb.problems.CustomProblem
   ctsb.problems.register_custom_problem


control
-------

.. toctree::
   :maxdepth: 3

.. autosummary::
  :toctree: _autosummary

   ctsb.problems.ControlProblem
   ctsb.problems.control.LDS_Control
   ctsb.problems.control.LSTM_Control
   ctsb.problems.control.RNN_Control
   ctsb.problems.control.CartPole
   ctsb.problems.control.DoublePendulum
   ctsb.problems.control.Pendulum


time_series
-----------

.. autosummary::
  :toctree: _autosummary

   ctsb.problems.TimeSeriesProblem
   ctsb.problems.time_series.SP500
   ctsb.problems.time_series.UCI_Indoor
   ctsb.problems.time_series.ENSO
   ctsb.problems.time_series.Crypto
   ctsb.problems.time_series.Random
   ctsb.problems.time_series.ARMA
   ctsb.problems.time_series.Unemployment
   ctsb.problems.time_series.LDS_TimeSeries
   ctsb.problems.time_series.LSTM_TimeSeries
   ctsb.problems.time_series.RNN_TimeSeries

pybullet
--------

.. autosummary::
  :toctree: _autosummary

  ctsb.problems.pybullet.PyBulletProblem
  ctsb.problems.pybullet.Simulator
  ctsb.problems.pybullet.Ant
  ctsb.problems.pybullet.CartPole
  ctsb.problems.pybullet.CartPoleDouble
  ctsb.problems.pybullet.CartPoleSwingup
  ctsb.problems.pybullet.HalfCheetah
  ctsb.problems.pybullet.Humanoid
  ctsb.problems.pybullet.Kuka
  ctsb.problems.pybullet.KukaDiverse
  ctsb.problems.pybullet.Minitaur
  ctsb.problems.pybullet.Obstacles
  

