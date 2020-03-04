class RegressionSystemID:
    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, K=None, learning_rate=0.001):
        self.initialized = True
        self.n, self.m = n, m
        self.T = 0
        self.K = K if K != None else np.zeros((m, n))
        self.lr = learning_rate
        self.stash = []
        self.x_history = []
        self.u_history = []

        # initialize matrices
        self.A = np.identity(n)
        self.B = np.zeros((n, m))

    def get_action(self, x_t, done):
        """ return action """
        self.T += 1
        # regular numpy
        eta_t = 1 - 2*random.randint(2, size=(self.m,))
        u_t = - self.K @ x_t + np.expand_dims(eta_t, axis=1)
        self.x_history.append(np.squeeze(x_t, axis=1))
        self.u_history.append(np.squeeze(u_t, axis=1))
        if done:
          if len(self.x_history) > 1:
            self.stash.append((self.x_history, self.u_history))
          self.x_history = []
          self.u_history = []
        return u_t

    def system_id(self):
        """ returns current estimate of hidden system dynamics """
        assert self.T > 1 # need at least 2 data points
        if len(self.x_history) > 1:
          self.stash.append((self.x_history, self.u_history))

        # transform x and u into regular numpy arrays for least squares
        x_t = onp.vstack([onp.array(x[:-1]) for x, u in self.stash])
        u_t = onp.vstack([onp.array(u[:-1]) for x, u in self.stash])
        x_t1 = onp.vstack([onp.array(x[1:]) for x, u in self.stash])

        # regression on A and B jointly
        A_B = scilinalg.lstsq(np.hstack((x_t, u_t)), x_t1)[0]
        A, B = np.array(A_B[:self.n]).T, np.array(A_B[self.n:]).T
        return (A, B)