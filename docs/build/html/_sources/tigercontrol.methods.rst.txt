tigercontrol.controllers package
===========================

.. automodule:: tigercontrol.controllers

core
----
.. autosummary::
  :toctree: _autosummary

   Controller

control
-------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.controllers.Controller
   tigercontrol.controllers.KalmanFilter
   tigercontrol.controllers.ODEShootingController
   tigercontrol.controllers.LQR
   tigercontrol.controllers.MPPI
   tigercontrol.controllers.CartPoleNN
   tigercontrol.controllers.ILQR


time_series
-----------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.controllers.time_series.TimeSeriesController
   tigercontrol.controllers.time_series.AutoRegressor
   tigercontrol.controllers.time_series.LastValue
   tigercontrol.controllers.time_series.PredictZero
   tigercontrol.controllers.time_series.RNN
   tigercontrol.controllers.time_series.LSTM
   tigercontrol.controllers.time_series.LeastSquares

optimizers
----------

.. autosummary::
  :toctree: _autosummary

   tigercontrol.utils.optimizers.Optimizer
   tigercontrol.utils.optimizers.Adagrad
   tigercontrol.utils.optimizers.Adam
   tigercontrol.utils.optimizers.ONS
   tigercontrol.utils.optimizers.SGD
   tigercontrol.utils.optimizers.OGD
   tigercontrol.utils.optimizers.mse
   tigercontrol.utils.optimizers.cross_entropy

boosting
--------

.. autosummary::
  :toctree: _autosummary

  tigercontrol.utils.boosting.SimpleBoost
