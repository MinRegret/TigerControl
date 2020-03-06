# Environment class
# Author: John Hallman

import tigercontrol
#from tigercontrol.controllers.quadcopter_controller import QuadcopterModel

def test_quadcopter():
    # Controller parameters
    CONTROLLER_PARAMETERS = {
        'Linear_PID':{'P':[300,300,7000],'I':[0.04,0.04,4.5],'D':[450,450,5000]},
        'Angular_PID':{'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]},
    }
    GOAL = (0,0,2)
    YAW = 0

    # initialize quadcopter and GUI
    quad_environment = tigercontrol.environment("Quadcopter")
    state = quad_environment.reset()

    # initialize model and targets
    quad_controller = QuadcopterModel()
    quad_controller.initialize(CONTROLLER_PARAMETERS, GOAL, YAW)

    # start control loop
    motor_action = None
    for i in range(1000):
        motor_action = quad_controller.get_action(state)
        state = quad_environment.step(motor_action)
    print("test_quadcopter passed")

if __name__ == "__main__":
    #test_quadcopter()
    pass
    
