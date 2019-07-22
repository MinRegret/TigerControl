ctsb.problems package
=====================

.. automodule:: ctsb.problems

core
----
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

   ctsb.problems.control.pybullet

.. autosummary::
  :toctree: _autosummary

   ctsb.problems.ControlProblem
   ctsb.problems.control.LDS
   ctsb.problems.control.LSTM_Output
   ctsb.problems.control.RNN_Output


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