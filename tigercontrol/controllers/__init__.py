# control init file

from tigercontrol.controllers.registration import controller_registry, controller_register, controller
from tigercontrol.controllers.custom import CustomController, register_custom_controller
from tigercontrol.controllers.core import Controller

# controllers
from tigercontrol.controllers.lqr import LQR
from tigercontrol.controllers.gpc import GPC
from tigercontrol.controllers.bpc import BPC
from tigercontrol.controllers.ilqr import ILQR

# boosting
from tigercontrol.controllers.boosting import DynaBoost


# ---------- Boosting Controllers ----------


controller_register(
    id='DynaBoost',
    entry_point='tigercontrol.controllers.boosting:DynaBoost',
)


# ---------- Control Controllers ----------


controller_register(
    id='LQR',
    entry_point='tigercontrol.controllers:LQR',
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
