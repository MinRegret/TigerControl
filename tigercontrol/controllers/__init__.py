# control init file

from tigercontrol.controllers.registration import controller_registry, controller_register, controller
from tigercontrol.controllers.custom import CustomController, register_custom_controller
from tigercontrol.controllers.core import Controller

# controllers
from tigercontrol.controllers.kalman_filter import KalmanFilter
from tigercontrol.controllers.shooting import Shooting
from tigercontrol.controllers.lqr import LQR
from tigercontrol.controllers.lqr_infinite_horizon import LQRInfiniteHorizon
from tigercontrol.controllers.ilqr import ILQR
from tigercontrol.controllers.gpc import GPC

# boosting
from tigercontrol.utils.boosting import SimpleBoost

# ---------- Boosting Controllers ----------


controller_register(
    id='SimpleBoost',
    entry_point='tigercontrol.utils.boosting:SimpleBoost',
)


# ---------- Control Controllers ----------


controller_register(
    id='KalmanFilter',
    entry_point='tigercontrol.controllers:KalmanFilter',
)

controller_register(
    id='Shooting',
    entry_point='tigercontrol.controllers:Shooting',
)

controller_register(
    id='LQR',
    entry_point='tigercontrol.controllers:LQR',
)

controller_register(
    id='LQRInfiniteHorizon',
    entry_point='tigercontrol.controllers:LQRInfiniteHorizon',
)

controller_register(
    id='ILQR',
    entry_point='tigercontrol.controllers:ILQR',
)

controller_register(
    id='GPC',
    entry_point='tigercontrol.controllers:GPC',
)

