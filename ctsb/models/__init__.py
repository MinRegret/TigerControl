# models init file

from ctsb.models.registration import model_registry, model_register, model
from ctsb.models.core import Model
from ctsb.models.custom import CustomModel, register_custom_model
from ctsb.models.optimizers import losses


# ---------- Time-Series Models ----------


model_register(
    id='LastValue',
    entry_point='ctsb.models.time_series:LastValue',
)

model_register(
    id='LeastSquares',
    entry_point='ctsb.models.time_series:LeastSquares',
)

model_register(
    id='AutoRegressor',
    entry_point='ctsb.models.time_series:AutoRegressor',
)

model_register(
    id='PredictZero',
    entry_point='ctsb.models.time_series:PredictZero',
)

model_register(
    id='RNN',
    entry_point='ctsb.models.time_series:RNN',
)

model_register(
    id='LSTM',
    entry_point='ctsb.models.time_series:LSTM',
)


# ---------- Boosting Models ----------


model_register(
    id='SimpleBoost',
    entry_point='ctsb.models.boosting:SimpleBoost',
)

model_register(
    id='SimpleBoostAdj',
    entry_point='ctsb.models.boosting:SimpleBoostAdj',
)


# ---------- Control Models ----------


model_register(
    id='KalmanFilter',
    entry_point='ctsb.models.control:KalmanFilter',
)

model_register(
    id='ODEShootingMethod',
    entry_point='ctsb.models.control:ODEShootingMethod',
)

model_register(
    id='LQR',
    entry_point='ctsb.models.control:LQR',
)

model_register(
    id='ILQR',
    entry_point='ctsb.models.control:ILQR',
)

model_register(
    id='MPPI',
    entry_point='ctsb.models.control:MPPI',
)

model_register(
    id='CartPoleNN',
    entry_point='ctsb.models.control:CartPoleNN',
)


