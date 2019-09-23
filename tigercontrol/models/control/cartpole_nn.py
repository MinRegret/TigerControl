# neural network policy class trained specifically for the cartpole problem
from tigercontrol.models.control.control_model import ControlModel
from tigercontrol.models.control.cartpole_weights import *

class CartPoleNN(ControlModel):
    ''' Description: Simple multi-layer perceptron policy, no internal state '''

    compatibles = set(['CartPole-v0', 'CartPoleSwingup-v0'])

    def __init__(self):
        self.initialized = False

    def initialize(self, observation_space, action_space):
        ''' Description: initialize the NN 
            Args:
                observation_space:
                action_space:
        '''
        self.initialized = True
        assert weights_dense1_w.shape == (observation_space[0], 64.0)
        assert weights_dense2_w.shape == (64.0, 32.0)
        assert weights_final_w.shape == (32.0, action_space[0])

    def predict(self, ob): # weights can be fount at the end of the file
        x = ob
        x = np.maximum((np.dot(x, weights_dense1_w) + weights_dense1_b), 0)
        x = np.maximum((np.dot(x, weights_dense2_w) + weights_dense2_b), 0)
        x = np.dot(x, weights_final_w) + weights_final_b
        return x