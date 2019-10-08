tigercontrol.methods package
===========================

.. automodule:: tigercontrol.methods

core
----
.. autosummary::
  :toctree: _autosummary

   Method

control
-------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.methods.control.ControlMethod
   tigercontrol.methods.control.KalmanFilter
   tigercontrol.methods.control.ODEShootingMethod
   tigercontrol.methods.control.LQR
   tigercontrol.methods.control.MPPI
   tigercontrol.methods.control.CartPoleNN
   tigercontrol.methods.control.ILQR


time_series
-----------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.methods.time_series.TimeSeriesMethod
   tigercontrol.methods.time_series.AutoRegressor
   tigercontrol.methods.time_series.LastValue
   tigercontrol.methods.time_series.PredictZero
   tigercontrol.methods.time_series.RNN
   tigercontrol.methods.time_series.LSTM
   tigercontrol.methods.time_series.LeastSquares

optimizers
----------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.methods.optimizers.Optimizer
   tigercontrol.methods.optimizers.Adagrad
   tigercontrol.methods.optimizers.Adam
   tigercontrol.methods.optimizers.ONS
   tigercontrol.methods.optimizers.SGD
   tigercontrol.methods.optimizers.OGD
   tigercontrol.methods.optimizers.mse
   tigercontrol.methods.optimizers.cross_entropy

boosting
--------

.. autosummary::
  :toctree: _autosummary

  tigercontrol.methods.boosting.SimpleBoost
