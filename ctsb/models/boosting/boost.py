# Boosting

import ctsb
import jax.numpy as np


def SimpleBoost(model, N = 10, timesteps = 1000):
    (model_id, model_params) = model

    ''' 1. Maintain N copies of the algorithm '''
    models = []
    for _ in range(N):
        new_model = ctsb.model(model_id)
        new_model.initialize()
        models.append(new_model)


