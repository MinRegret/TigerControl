""" Quadcopter environment implementation """

import numpy as onp
import jax.numpy as np
import jax
import jax.random as random
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

from tigercontrol.environments import Environment
from tigercontrol.utils import generate_key


class Quadcopter(Environment):
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
        self.map_width = 4.0
        self.map_height = 8.0
        self.dt = 0.01 # seconds
        # From Quadrotor Dynamics and Control by Randal Beard
        ixx=((2*self.weight*self.yaw_rate**2)/5)+(2*self.weight*self.L**2)
        iyy=ixx
        izz=((2*self.weight*self.yaw_rate**2)/5)+(4*self.weight*self.L**2)
        self.I = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
        self.invI = np.linalg.inv(self.I)
        self.render_init = False

        @jax.jit
        def _dynamics(s, a):
            thrusts = a * 4.392e-8 * np.power(self.prop_dia, 3.5)/(np.sqrt(self.prop_pitch))
            thrusts = thrusts * (4.23e-4 * a * self.prop_pitch)
            s_new = s + self.dt * self._state_dot(s, thrusts)
            s_new = jax.ops.index_update(s_new, jax.ops.index[6:9], self._wrap_angle(s_new[6:9]))
            s_new = jax.ops.index_update(s_new, 2, np.max((0.0, s_new[2]))) # z >= 0 
            return s_new
        self._dynamics = _dynamics

    def reset(self):
        # random initial state
        xy = random.uniform(generate_key(), minval=-self.map_width, maxval=self.map_width, shape=(2,))
        z = random.uniform(generate_key(), minval=1.0, maxval=self.map_height, shape=(1,))
        init_pos = np.concatenate((xy, z))
        init_pos = np.array([-3, -3, 7]) # constant starting position for debugging
        self.state = np.concatenate((init_pos, np.zeros(9)))
        self._render()
        return self.state

    def step(self, motor_action):
        self.state = self._dynamics(self.state, motor_action)
        self._render()
        return self.state

    def _rotation_matrix(self, angles):
        ct, cp, cg = np.cos(angles)
        st, sp, sg = np.sin(angles)
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def _wrap_angle(self,val):
        return (val + np.pi) % (2 * np.pi) - np.pi 

    def _state_dot(self, s, thrusts):
        t1, t2, t3, t4 = thrusts
        x_dot = s[3:6] # The velocities(t+1 x_dots equal the t x_dots)
        R = self._rotation_matrix(s[6:9]) # The acceleration
        x_dotdot = np.array([0,0,-self.weight*self.g])+np.dot(R, np.array([0, 0, np.sum(thrusts)]))/self.weight
        a_dot = s[9:12] # The angular rates(t+1 theta_dots equal the t theta_dots)
        # The angular accelerations
        tau = np.array([self.L*(t1-t3), self.L*(t2-t4), self.b*(t1-t2+t3-t4)])
        a_dotdot = np.dot(self.invI, (tau - np.cross(a_dot, np.dot(self.I, a_dot))))
        state_dot = np.concatenate((x_dot, x_dotdot, a_dot, a_dotdot))
        return state_dot

    def _render(self):
        if not self.render_init:
            self._init_render()
            self.render_init = True
        R = self._rotation_matrix(self.state[6:9])
        L = self.L
        points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
        points = np.dot(R,points)
        points = jax.ops.index_add(points, jax.ops.index[0,:], self.state[0])
        points = jax.ops.index_add(points, jax.ops.index[1,:], self.state[1])
        points = jax.ops.index_add(points, jax.ops.index[2,:], self.state[2])
        points = onp.array(points) # jax numpy works poorly with matplotlib 
        
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
        self.ax.set_xlim3d([-self.map_width, self.map_width])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-self.map_width, self.map_width])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([0, self.map_height])
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadcopter Simulation')
        self.l1, = self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)
        self.l2, = self.ax.plot([],[],[],color='red',linewidth=3,antialiased=False)
        self.hub, = self.ax.plot([],[],[],marker='o',color='green', markersize=6,antialiased=False)



