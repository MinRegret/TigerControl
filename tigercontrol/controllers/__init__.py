# control init file

from tigercontrol.controllers.registration import controller_registry, controller_register, controller
from tigercontrol.controllers.custom import CustomController, register_custom_controller
from tigercontrol.controllers.core import Controller

# controllers
from tigercontrol.controllers.lqr import LQR
from tigercontrol.controllers.gpc import GPC
from tigercontrol.controllers.gpc_v1 import GPC_v1
from tigercontrol.controllers.gpc_v2 import GPC_v2
from tigercontrol.controllers.bpc import BPC
from tigercontrol.controllers.kalman_filter import KalmanFilter
from tigercontrol.controllers.shooting import Shooting
from tigercontrol.controllers.ilqr import ILQR

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
    id='LQRFiniteHorizon',
    entry_point='tigercontrol.controllers:LQRFiniteHorizon',
)

controller_register(
    id='ILQR',
    entry_point='tigercontrol.controllers:ILQR',
)

controller_register(
    id='GPC',
    entry_point='tigercontrol.controllers:GPC',
)

controller_register(
    id='BPC',
    entry_point='tigercontrol.controllers:BPC',
)

controller_register(
    id='GPC-v1',
    entry_point='tigercontrol.controllers:GPC_v1',
)


controller_register(
    id='GPC-v2',
    entry_point='tigercontrol.controllers:GPC_v2',
)