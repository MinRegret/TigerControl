# methods init file

from tigercontrol.methods.registration import method_registry, method_register, method
from tigercontrol.methods.core import Method
from tigercontrol.methods.custom import CustomMethod, register_custom_method
from tigercontrol.methods.optimizers import losses


# ---------- Time-Series Methods ----------

method_register(
    id='WaveFiltering',
    entry_point='tigercontrol.methods.time_series:WaveFiltering',
)

method_register(
    id='LastValue',
    entry_point='tigercontrol.methods.time_series:LastValue',
)

method_register(
    id='LeastSquares',
    entry_point='tigercontrol.methods.time_series:LeastSquares',
)

method_register(
    id='AutoRegressor',
    entry_point='tigercontrol.methods.time_series:AutoRegressor',
)

method_register(
    id='PredictZero',
    entry_point='tigercontrol.methods.time_series:PredictZero',
)

method_register(
    id='RNN',
    entry_point='tigercontrol.methods.time_series:RNN',
)

method_register(
    id='LSTM',
    entry_point='tigercontrol.methods.time_series:LSTM',
)


# ---------- Boosting Methods ----------


method_register(
    id='SimpleBoost',
    entry_point='tigercontrol.methods.boosting:SimpleBoost',
)

method_register(
    id='SimpleBoostAdj',
    entry_point='tigercontrol.methods.boosting:SimpleBoostAdj',
)


# ---------- Control Methods ----------


method_register(
    id='KalmanFilter',
    entry_point='tigercontrol.methods.control:KalmanFilter',
)

method_register(
    id='ODEShootingMethod',
    entry_point='tigercontrol.methods.control:ODEShootingMethod',
)

method_register(
    id='LQR',
    entry_point='tigercontrol.methods.control:LQR',
)

method_register(
    id='ILQR',
    entry_point='tigercontrol.methods.control:ILQR',
)

method_register(
    id='MPPI',
    entry_point='tigercontrol.methods.control:MPPI',
)

method_register(
    id='CartPoleNN',
    entry_point='tigercontrol.methods.control:CartPoleNN',
)


