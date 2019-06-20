# models init file

from ctsb.models.registration import model_registry, model_register, model, model_spec
from ctsb.models.core import Model


# ---------- Models ----------


model_register(
    id='Linear',
    entry_point='ctsb.models.time_series:Linear',
)

