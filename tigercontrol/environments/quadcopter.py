""" Quadcopter environment implementation """

#import jax.numpy as np
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from tigercontrol.environments import Environment


class Quadcopter(Environment):
    class Propeller(): # propeller helper class 
        def __init__(self, prop_dia, prop_pitch):
            self.dia = prop_dia
            self.pitch = prop_pitch
            self.speed = 0 #RPM
            self.thrust = 0

        def set_speed(self, speed):
            self.speed = speed
            # From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
            self.thrust = 4.392e-8 * self.speed * np.power(self.dia,3.5)/(np.sqrt(self.pitch))
            self.thrust = self.thrust*(4.23e-4 * self.speed * self.pitch)

    # State space representation: [x y z x_dot y_dot z_dot theta phi gamma theta_dot phi_dot gamma_dot]
    # From Quadcopter Dynamics, Simulation, and Control by Andrew Gibiansky
    def __init__(self):
        self.g = 9.81 # default gravity
        self.b = 0.0245 # default inertia
        self.weight = 1.2 # N or kg?
        self.yaw_rate = 0.1 
        self.L = 0.3 # quadcopter width/length
        self.prop_dia = 10.0 # propeller diameter
        self.prop_pitch = 4.5 # propeller pitch
        self.map_width = 2.0
        self.map_height = 6.0
        self.ode = scipy.integrate.ode(self._state_dot).set_integrator('vode', nsteps=5, method='bdf')
        self.prop1 = self.Propeller(self.prop_dia, self.prop_pitch)
        self.prop2 = self.Propeller(self.prop_dia, self.prop_pitch)
        self.prop3 = self.Propeller(self.prop_dia, self.prop_pitch)
        self.prop4 = self.Propeller(self.prop_dia, self.prop_pitch)
        # From Quadrotor Dynamics and Control by Randal Beard
        ixx=((2*self.weight*self.yaw_rate**2)/5)+(2*self.weight*self.L**2)
        iyy=ixx
        izz=((2*self.weight*self.yaw_rate**2)/5)+(4*self.weight*self.L**2)
        self.I = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
        self.invI = np.linalg.inv(self.I)
        init_pos = np.concatenate((np.random.uniform(-self.map_width, self.map_width, size=(2,)), np.random.uniform(0,self.map_height, size=(1,))))
        self.state = np.concatenate((init_pos, np.zeros(9))) # initial state
        self.render_init = False

    def reset(self):
        self.QUAD_DYNAMICS_UPDATE = 0.010 # seconds
        self.render()
        return self.state

    def step(self, motor_action):
        self.prop1.set_speed(motor_action[0])
        self.prop2.set_speed(motor_action[1])
        self.prop3.set_speed(motor_action[2])
        self.prop4.set_speed(motor_action[3])
        self.update(self.QUAD_DYNAMICS_UPDATE)
        self.render()
        return self.state

    def update(self, dt):
        self.ode.set_initial_value(self.state,0).set_f_params('q1')
        self.state = self.ode.integrate(dt)
        self.state[6:9] = self._wrap_angle(self.state[6:9]) # angles
        self.state[2] = max(0,self.state[2]) # cannot go below floor level

    def render(self):
        if not self.render_init:
            self._init_render()
            self.render_init = True
        R = self._rotation_matrix(self.state[6:9])
        L = self.L
        points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
        points = np.dot(R,points)
        points[0,:] += self.state[0]
        points[1,:] += self.state[1]
        points[2,:] += self.state[2]
        self.l1.set_data(points[0,0:2],points[1,0:2])
        self.l1.set_3d_properties(points[2,0:2])
        self.l2.set_data(points[0,2:4],points[1,2:4])
        self.l2.set_3d_properties(points[2,2:4])
        self.hub.set_data(points[0,5],points[1,5])
        self.hub.set_3d_properties(points[2,5])
        plt.pause(0.00001)

    def _init_render(self):
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([-2.0, 2.0])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-2.0, 2.0])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([0, 5.0])
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadcopter Simulation')
        self.l1, = self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)
        self.l2, = self.ax.plot([],[],[],color='red',linewidth=3,antialiased=False)
        self.hub, = self.ax.plot([],[],[],marker='o',color='green', markersize=6,antialiased=False)

    def _rotation_matrix(self, angles):
        ct = np.cos(angles[0])
        cp = np.cos(angles[1])
        cg = np.cos(angles[2])
        st = np.sin(angles[0])
        sp = np.sin(angles[1])
        sg = np.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def _wrap_angle(self,val):
        return( ( val + np.pi) % (2 * np.pi ) - np.pi )

    def _state_dot(self, time, state):
        state_dot = np.zeros(12)
        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0] = self.state[3]
        state_dot[1] = self.state[4]
        state_dot[2] = self.state[5]
        # The acceleration
        x_dotdot = np.array([0,0,-self.weight*self.g]) + np.dot(self._rotation_matrix(self.state[6:9]), \
            np.array([0,0,(self.prop1.thrust + self.prop2.thrust + self.prop3.thrust + self.prop4.thrust)])) /self.weight
        state_dot[3] = x_dotdot[0]
        state_dot[4] = x_dotdot[1]
        state_dot[5] = x_dotdot[2]
        # The angular rates(t+1 theta_dots equal the t theta_dots)
        state_dot[6] = self.state[9]
        state_dot[7] = self.state[10]
        state_dot[8] = self.state[11]
        # The angular accelerations
        omega = self.state[9:12]
        tau = np.array([self.L*(self.prop1.thrust-self.prop3.thrust), self.L*(self.prop2.thrust-self.prop4.thrust), \
            self.b*(self.prop1.thrust-self.prop2.thrust+self.prop3.thrust-self.prop4.thrust)])
        omega_dot = np.dot(self.invI, (tau - np.cross(omega, np.dot(self.I,omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot



