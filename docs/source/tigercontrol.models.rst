tigercontrol.models package
=====================

.. automodule:: tigercontrol.models

core
----
.. autosummary::
  :toctree: _autosummary

   Model

control
-------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.models.control.ControlModel
   tigercontrol.models.control.KalmanFilter
   tigercontrol.models.control.ODEShootingMethod
   tigercontrol.models.control.LQR
   tigercontrol.models.control.MPPI
   tigercontrol.models.control.CartPoleNN
   tigercontrol.models.control.AdversarialDisturbances
   tigercontrol.models.control.ILQR


time_series
-----------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.models.time_series.TimeSeriesModel
   tigercontrol.models.time_series.AutoRegressor
   tigercontrol.models.time_series.LastValue
   tigercontrol.models.time_series.PredictZero
   tigercontrol.models.time_series.rnn.RNN
   tigercontrol.models.time_series.lstm.LSTM
   tigercontrol.models.time_series.least_squares.LeastSquares

optimizers
----------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.models.optimizers.Optimizer
   tigercontrol.models.optimizers.Adagrad
   tigercontrol.models.optimizers.Adam
   tigercontrol.models.optimizers.ONS
   tigercontrol.models.optimizers.SGD
   tigercontrol.models.optimizers.OGD
   tigercontrol.models.optimizers.mse
   tigercontrol.models.optimizers.cross_entropy

boosting
--------

.. autosummary::
  :toctree: _autosummary

  tigercontrol.models.boosting.SimpleBoost
