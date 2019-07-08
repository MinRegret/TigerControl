# models init file

from ctsb.models.registration import model_registry, model_register, model, model_spec
from ctsb.models.core import Model
from ctsb.models.custom import CustomModel, register_custom_model


# ---------- Models ----------


model_register(
    id='LastValue',
    entry_point='ctsb.models.time_series:LastValue',
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
    id='KalmanFilter',
    entry_point='ctsb.models.control:KalmanFilter',
)

model_register(
    id='ShootingMethod',
    entry_point='ctsb.models.control:ShootingMethod',
)