# control init file

from tigercontrol.planners.registration import planner_registry, planner_register, planner
from tigercontrol.planners.custom import CustomPlanner, register_custom_planner
from tigercontrol.planners.core import Planner

# planners
from tigercontrol.planners.ilqr import ILQR


# ---------- Control Planners ----------


planner_register(
    id='ILQR',
    entry_point='tigercontrol.planners:ILQR',
)
