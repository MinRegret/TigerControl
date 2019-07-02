# models init file

from ctsb.models.registration import model_registry, model_register, model, model_spec
from ctsb.models.core import Model
from ctsb.models.custom_model import CustomModel


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
