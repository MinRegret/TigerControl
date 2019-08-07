ctsb.models package
=====================

.. automodule:: ctsb.models

core
----
.. autosummary::
  :toctree: _autosummary

   Model

control
-------

.. autosummary::
  :toctree: _autosummary

   ctsb.models.control.ControlModel
   ctsb.models.control.KalmanFilter
   ctsb.models.control.ODEShootingMethod
   ctsb.models.control.LQR
   ctsb.models.control.MPPI
   ctsb.models.control.CartPoleNN
   ctsb.models.control.AdversarialDisturbances
   ctsb.models.control.ILQR


time_series
-----------

.. autosummary::
  :toctree: _autosummary

   ctsb.models.time_series.TimeSeriesModel
   ctsb.models.time_series.AutoRegressor
   ctsb.models.time_series.LastValue
   ctsb.models.time_series.PredictZero
   ctsb.models.time_series.rnn.RNN
   ctsb.models.time_series.lstm.LSTM
   ctsb.models.time_series.least_squares.LeastSquares

optimizers
----------

.. autosummary::
  :toctree: _autosummary

   ctsb.models.optimizers.Optimizer
   ctsb.models.optimizers.Adagrad
   ctsb.models.optimizers.Adam
   ctsb.models.optimizers.ONS
   ctsb.models.optimizers.SGD
   ctsb.models.optimizers.OGD
   ctsb.models.optimizers.mse
   ctsb.models.optimizers.cross_entropy

boosting
--------

.. autosummary::
  :toctree: _autosummary

  ctsb.models.boosting.SimpleBoost
