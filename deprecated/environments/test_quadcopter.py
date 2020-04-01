# Environment class
# Author: John Hallman

import tigercontrol
from tigercontrol.controllers import Controller
import jax
import jax.numpy as np


# run test
def test_quadcopter():
    # Controller parameters
    CONTROLLER_PARAMETERS = {
        'Linear_PID':{'P':np.array([300,300,7000]),'I':np.array([0.04,0.04,4.5]),'D':np.array([450,450,5000])},
        'Angular_PID':{'P':np.array([22000,22000,1500]),'I':np.array([0,0,1.2]),'D':np.array([12000,12000,0])},
    }
    GOAL = np.array([0, 0, 2]) # x, y, z
    YAW = 0

    # initialize quadcopter and GUI
    quad_environment = tigercontrol.environment("Quadcopter")
    #quad_environment = quad_environment_class()
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



class QuadcopterModel(Controller):
    ''' Description: Simple multi-layer perceptron policy, no internal state '''

    def __init__(self):
        self.initialized = False

    def initialize(self, params, goal, yaw):
        self.MOTOR_LIMITS = [4000, 9000]
        self.TILT_LIMITS = [(-10/180.0)*3.14,(10/180.0)*3.14]
        self.YAW_CONTROL_LIMITS = [-900,900]
        self.LINEAR_TO_ANGULAR_SCALER = [1,1,0]
        self.YAW_RATE_SCALER = 0.18
        self.Z_LIMITS = [self.MOTOR_LIMITS[0]+500,self.MOTOR_LIMITS[1]-500]

        self.LINEAR_P = params['Linear_PID']['P']
        self.LINEAR_I = params['Linear_PID']['I']
        self.LINEAR_D = params['Linear_PID']['D']
        self.ANGULAR_P = params['Angular_PID']['P']
        self.ANGULAR_I = params['Angular_PID']['I']
        self.ANGULAR_D = params['Angular_PID']['D']

        self.p_i_term = np.zeros(3)
        self.a_i_term = np.zeros(3)
        self.thetai_term = 0
        self.phii_term = 0
        self.gammai_term = 0
        self.thread_object = None
        self.target = goal
        self.yaw_target = self._wrap_angle(yaw)

        @jax.jit
        def _get_action(obs, p_i_term, a_i_term):
            p, p_dot, a, a_dot = obs[:3], obs[3:6], obs[6:9], obs[9:12]
            theta, phi, gamma = a
            theta_dot, phi_dot, gamma_dot = a_dot

            err = self.target - p
            p_i_term += self.LINEAR_I * err
            dest_dot = self.LINEAR_P * err - self.LINEAR_D * p_dot + p_i_term
            dest_x_dot, dest_y_dot, dest_z_dot = dest_dot
            throttle = np.clip(dest_z_dot, self.Z_LIMITS[0], self.Z_LIMITS[1])

            dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0]*(dest_x_dot*np.sin(gamma)-dest_y_dot*np.cos(gamma))
            dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1]*(dest_x_dot*np.cos(gamma)+dest_y_dot*np.sin(gamma))
            dest_gamma = self.yaw_target
            dest_theta = np.clip(dest_theta,self.TILT_LIMITS[0],self.TILT_LIMITS[1])
            dest_phi = np.clip(dest_phi,self.TILT_LIMITS[0],self.TILT_LIMITS[1])
            theta_error = dest_theta-theta
            phi_error = dest_phi-phi
            gamma_dot_error = (self.YAW_RATE_SCALER*self._wrap_angle(dest_gamma-gamma)) - gamma_dot

            a_err = np.array([theta_error, phi_error, gamma_dot_error])
            a_i_term += self.ANGULAR_I * a_err
            p_val = self.ANGULAR_P * a_err - self.ANGULAR_D * a_dot + a_i_term
            x_val, y_val, z_val = p_val
            z_val = np.clip(z_val, self.YAW_CONTROL_LIMITS[0],self.YAW_CONTROL_LIMITS[1])

            m1 = throttle + x_val + z_val
            m2 = throttle + y_val - z_val
            m3 = throttle - x_val + z_val
            m4 = throttle - y_val - z_val
            M = np.array([m1,m2,m3,m4])
            M = np.clip(M, self.MOTOR_LIMITS[0], self.MOTOR_LIMITS[1])
            return M, p_i_term, a_i_term
        self._get_action = _get_action

    def _wrap_angle(self,val):
        return (val + np.pi) % (2 * np.pi) - np.pi

    def get_action(self, obs): # weights can be fount at the end of the file
        a, self.p_i_term, self.a_i_term = self._get_action(obs, self.p_i_term, self.a_i_term)
        return a

if __name__ == "__main__":
    test_quadcopter()
    
