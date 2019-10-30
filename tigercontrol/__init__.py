# controllers init file

from tigercontrol.controllers.registration import controller_registry, controller_register, controller
from tigercontrol.controllers.core import Controller
from tigercontrol.controllers.custom import CustomController, register_custom_controller
from tigercontrol.utils.optimizers import losses



# ---------- Boosting Controllers ----------


controller_register(
    id='SimpleBoost',
    entry_point='tigercontrol.utils.boosting:SimpleBoost',
)

controller_register(
    id='SimpleBoostAdj',
    entry_point='tigercontrol.utils.boosting:SimpleBoostAdj',
)


# ---------- Control Controllers ----------


controller_register(
    id='KalmanFilter',
    entry_point='tigercontrol.controllers.control:KalmanFilter',
)

controller_register(
    id='ODEShootingMethod',
    entry_point='tigercontrol.controllers.control:ODEShootingMethod',
)

controller_register(
    id='LQR',
    entry_point='tigercontrol.controllers.control:LQR',
)

controller_register(
    id='ILQR',
    entry_point='tigercontrol.controllers.control:ILQR',
)

controller_register(
    id='CartPoleNN',
    entry_point='tigercontrol.controllers.control:CartPoleNN',
)

controller_register(
    id='GPC',
    entry_point='tigercontrol.controllers.control:GPC',
)

