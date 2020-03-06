"""
Double pendulum environment, taken from OpenAI Gym
"""
import jax
import jax.numpy as np
import jax.random as random
from jax.numpy import sin, cos

from tigercontrol.utils import generate_key
from tigercontrol.environments import Environment

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class DoublePendulum(Environment):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    dt = .2

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    torque_noise_max = 0.
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self):
        self.initialized = False
        self.viewer = None
        #high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
        self.observation_space = (6,)
        self.action_space = (1,)
        self.state = None
        def wrap(x):
            x = np.where(x > np.pi, x - 2*np.pi, x)
            x = np.where(x < -np.pi, x + 2*np.pi, x)
            return x

        def _dynamics(x, u):
            x, u = np.squeeze(x, axis=1), np.squeeze(u, axis=1)
            torque = np.clip(u, -1.0, 1.0)[0]
            s_augmented = np.append(x, torque)
            ns = self._rk4(self._dsdt, s_augmented, [0, self.dt])
            ns_0 = wrap(ns[0])
            ns_1 = wrap(ns[1])
            ns_2 = np.clip(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
            ns_3 = np.clip(ns[3], -self.MAX_VEL_1, self.MAX_VEL_1)
            return np.expand_dims(np.array([ns_0, ns_1, ns_2, ns_3]), axis=1)
        self._dynamics = jax.jit(dynamics)

    def reset(self):
        self.state = random.uniform(generate_key(), minval=-0.1, maxval=0.1, shape=(4,))
        return self.state


    def _rk4(self, derivs, y0, t):
        """(self._dsdt, s_augmented, [0, self.dt])
        Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
        """
        yout = y0
        for i in range(len(t) - 1):
            thist = t[i]
            dt = t[i + 1] - thist
            dt2 = dt / 2.0
            k1 = np.asarray(derivs(yout, thist))
            k2 = np.asarray(derivs(yout + dt2 * k1, thist + dt2))
            k3 = np.asarray(derivs(yout + dt2 * k2, thist + dt2))
            k4 = np.asarray(derivs(yout + dt * k3, thist + dt))
            yout = yout + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return yout[:4]

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
            / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.])

    def step(self, a):
        self.state = self._dynamics(self.state, a)
        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        return self.state, reward, terminal, {}

    def _terminal(self):
        s = self.state
        return bool(- cos(s[0]) - cos(s[1] + s[0]) > 1.)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None

        p1 = [-self.LINK_LENGTH_1 *
              np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]-np.pi/2, s[0]+s[1]-np.pi/2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



