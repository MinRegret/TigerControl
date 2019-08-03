"""
Linear Quadratic Regulator
"""
import jax
import jax.numpy as np
import ctsb
from ctsb.models.control import ControlModel

class iLQR(ControlModel):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """
    
    compatibles = set([])

    def __init__(self):
        self.initialized = False

    def to_ndarray(self, x):
        """
        Description: If x is a scalar, transform it to a (1, 1) numpy.ndarray;
        otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
        Returns:
            A numpy.ndarray representation of x
        """
        x = np.asarray(x)
        if(np.ndim(x) == 0):
            x = x[None, None]
        return x

    def extend(self, x, T):
        """
        Description: If x is not in the correct form, convert it; otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
            T (postive int): number of timesteps
        Returns:
            A numpy.ndarray representation of x
        """
        x = self.to_ndarray(x)
        return [x for i in range(T)]

    def initialize(self, F, f, C, c, T, x):
        """
        Description: Initialize the dynamics of the model
        Args:
            F (float/numpy.ndarray): past value contribution coefficients
            f (float/numpy.ndarray): bias coefficients
            C (float/numpy.ndarray): quadratic cost coefficients
            c (float/numpy.ndarray): linear cost coefficients
            T (postive int): number of timesteps
            x (float/numpy.ndarray): initial state
        """
        self.initialized = True
        
        self.F, self.f, self.C, self.c, self.T, self.x = self.extend(F, T), self.extend(f, T), self.extend(C, T), self.extend(c, T), T, self.to_ndarray(x)
        
        self.u = self.extend(np.zeros((self.F[0].shape[1] - self.x.shape[0], 1)), T)
        
        self.K = self.extend(np.zeros((self.u[0].shape[0], self.x.shape[0])), T)
        self.k = self.u.copy()

    def ilqr(self, u_shape, dyn, L, x_0, T, threshold, lamb=0.1):
        # initialize
        u = self.extend(np.zeros((u_shape, 1)), T)
        # print("u.shape : " + str(u[0].shape))

        count = 0
        while True:
            count += 1
            # print(str(u))
            x = [x_0]
            # print("x_0 : " + str(x_0))
            # print("u: " + str(u))
            for i in range(T):
                # print("i = " + str(i))
                # print("x : " + str(x))
                # print("x[-1] : " + str(x[-1]))
                x.append(dyn(x[-1],u[i]))
            # print("x[-1] : " + str(x[-1]))
            F = [jax.jacrev(dyn, argnums=(0,1))(x[t],u[t][0]) for t in range(T)]
            # print("F[0] : " + str(F[0]))
            F = [np.hstack([F_x, F_u]) for (F_x, F_u) in F]
            # print("self.to_ndarray(x[0] : " + str(self.to_ndarray(x[0])))
            # print("self.to_ndarray(u[0] : " + str(u[0]))

            # print("F[0]: " + str(F[0]))
            # print("F[0].shape : " + str(F[0].shape))
            # print("np.concatenate((self.to_ndarray(x[0]), u[0][0])).shape : " + str(np.concatenate((self.to_ndarray(x[0]), u[0][0])).shape))
            f = [dyn(x[t],u[t]) - F[t] @ np.concatenate((self.to_ndarray(x[t]), u[t][0])) for t in range(T)]
            f = [np.asarray([[y] for y in f_i]) for f_i in f]
            # print("dyn(x[t],u[t]) : " + str(dyn(x[0],u[0])))
            # print("dyn(x[t],u[t]).T : " + str(dyn(x[0],u[0]).T))
            # print("f[0] : " + str(f[0]))

            C = [jax.hessian(L, argnums=(0,1))(x[t],u[t][0]) for t in range(T)]
            # print("C[0]:" + str(C[0]))
            C = [np.vstack([np.hstack([C_x[0],C_x[1]]), np.hstack([C_u[0],C_u[1]])]) for (C_x, C_u) in C]
            # print("C[0]:" + str(C[0]))
            c = [jax.grad(L, argnums=(0,1))(x[t],u[t][0]) for t in range(T)]
            # print("c[0] before : " +str(c[0]))
            c = [np.vstack([np.asarray([[y] for y in c_t[0]]), np.asarray(c_t[1])]) for c_t in c]
            
            # print("c[0] after : " +str(c[0]))

            x_0_col = np.asarray([[y] for y in x_0])
            u_new = self.lqr(F, f, C, c, x_0_col, x, T, lamb)
            # print("u_new = " + str(u_new))

            x_new = [x_0]
            for i in range(T):
                x_new.append(dyn(x_new[-1], u_new[i]))
            # print("len(x_new) : " + str(len(x_new)))

            old_cost = np.sum([L(x[t],u[t]) for t in range(T)])
            new_cost = np.sum([L(x_new[t],u_new[t]) for t in range(T)])
            # print("u_new : " + str(u_new))
            print("old_cost : " + str(old_cost))
            print("new_cost : " + str(new_cost))
            print("lamb : " + str(lamb))

            if new_cost < old_cost:
                u = u_new
                lamb = 2.0 * lamb
            else:
                lamb = 0.5 *lamb
            
            if abs(new_cost - old_cost) < threshold:
                break;

        return u

    def lqr(self, F, f, C, c, x, xs, T, lamb):
        """
        Description: Updates internal parameters and then returns the estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions
        """
        u = self.extend(np.zeros((F[0].shape[1] - x.shape[0], 1)), T)
        K = self.extend(np.zeros((u[0].shape[0], x.shape[0])), T)
        k = u.copy()
        ## Initialize V and Q Functions ##
        V = np.zeros((F[0].shape[0], F[0].shape[0]))
        v = np.zeros((F[0].shape[0], 1))
        Q = np.zeros((C[0].shape[0], C[0].shape[1]))
        q = np.zeros((c[0].shape[0], 1))

        # print("V : " + str(V))
        # print("C[0] : " + str(C[0]))
        # print("q : " + str(q))
        # print("c[4]: " + str(c[4]))
        # print("F[4].T : " + str(F[4].T))
        # print("f[4] : " + str(f[4]))
        # print("F[4].T @ V @ f[4] : " + str(F[4].T @ V @ f[4]))
        # print("F[4].T @ v: " + str(F[4].T @ v))
        ## Backward Recursion ##
        for t in range(T - 1, -1, -1):
            # print ("--------------------------------------------- t : " + str(t))

            Q = C[t] + F[t].T @ V @ F[t]
            q = c[t] + F[t].T @ V @ f[t] + F[t].T @ v

            # print("C[t] : " + str(C[t]))
            # print("F[t] : " + str(F[t]))
            # print("V : " + str(V))
            Q_uu = Q[x.shape[0] :, x.shape[0] :]
            # print("Q_uu : " + str(Q_uu))
            Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
            if t == 0: print("Q_uu_evals before: " + str(Q_uu_evals))
            Q_uu_evals = [(ev if ev > 0 else 0) + lamb for ev in Q_uu_evals]
            if t == 0: print("Q_uu_evals after: " + str(Q_uu_evals))
            Q_uu_inv = Q_uu_evecs @ np.diag(np.asarray([1./ev for ev in Q_uu_evals])) @ Q_uu_evecs.T
            Q_uu = Q_uu_evecs @ np.diag(np.asarray([ev for ev in Q_uu_evals])) @ Q_uu_evecs.T

            K[t] = -Q_uu_inv @ Q[x.shape[0] :, : x.shape[0]]
            k[t] = -Q_uu_inv @ q[x.shape[0] :]

            # print("Q_uu_inv : " + str(Q_uu_inv))
            # print("q: " + str(q))
            # print("c[t]: " + str(c[t]))
            # print("c[t].shape : " + str(c[t].shape))
            # print("F[t].T.shape : " + str(F[t].T.shape))
            # print("V.shape : " + str(V.shape))
            # print("f[t].shape : " +str(f[t].shape))

            # print("v.shape: " + str(v.shape))
            # print("q[x.shape[0] :] : " + str(q[x.shape[0] :]))
            # print("x.shape[0] : " + str(x.shape[0]))

            K[t] = K[t].astype(float)
            k[t] = k[t].astype(float)

            # print("K[t] : " + str(K[t]))
            # print("type(K[t]) : "  + str(K[t].dtype))
            # print("type(V) : " + str(V.dtype))
            # print("k[t] : " + str(k[t]))

            V = Q[: x.shape[0], : x.shape[0]] + Q[: x.shape[0], x.shape[0] :] @ K[t] + K[t].T @ Q[x.shape[0] :, : x.shape[0]] + K[t].T @ Q[x.shape[0] :, x.shape[0] :] @ K[t]
            v = q[: x.shape[0]] + Q[: x.shape[0], x.shape[0] :] @ k[t] + K[t].T @ q[x.shape[0] :] + K[t].T @ Q[x.shape[0] :, x.shape[0] :] @ k[t]

            # V = Q[: x.shape[0], : x.shape[0]] + Q[: x.shape[0], x.shape[0] :] @ K[t] + K[t].T @ Q[x.shape[0] :, : x.shape[0]] + K[t].T @ Q_uu @ K[t]
            # v = q[: x.shape[0]] + Q[: x.shape[0], x.shape[0] :] @ k[t] + K[t].T @ q[x.shape[0] :] + K[t].T @ Q_uu @ k[t]

        ## Forward Recursion ##
        xs = [np.asarray([[y] for y in x]) for x in xs]
        x_new = [xs[0]] + [None for i in range(T-1)]
        u_new = [0 for i in range(T)]
        for t in range(T):
            # print("xs[t] = " + str(xs[t]))
            # print("x_new[t] = " + str(x_new[t]))
            # print("K[t] : " + str(K[t]))
            # print("k[t] : " + str(k[t]))
            # print("u[t] = " + str(u[t]))
            u_new[t] = u[t] + k[t] + K[t] @ (x_new[t] - xs[t])            
            if t < T-1:
                x_new[t+1]= F[t] @ np.vstack((x_new[t], u_new[t])) + f[t]

        return u_new

    def predict(self):
        """
        Description: Returns estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions
        """
        return self.u


    def update(self):
        """
        Description: Updates internal parameters
        Args:
            None
        """

        ## Initialize V and Q Functions ##
        V = np.zeros((self.F[0].shape[0], self.F[0].shape[0]))
        v = np.zeros((self.F[0].shape[0], 1))
        Q = np.zeros((self.C[0].shape[0], self.C[0].shape[1]))
        q = np.zeros((self.c[0].shape[0], 1))

        ## Backward Recursion ##
        for t in range(self.T - 1, -1, -1):

            Q = self.C[t] + self.F[t].T @ V @ self.F[t]
            q = self.c[t] + self.F[t].T @ V @ self.f[t] + self.F[t].T @ v

            self.K[t] = -np.linalg.inv(Q[self.x.shape[0] :, self.x.shape[0] :]) @ Q[self.x.shape[0] :, : self.x.shape[0]]
            self.k[t] = -np.linalg.inv(Q[self.x.shape[0] :, self.x.shape[0] :]) @ q[self.x.shape[0] :]

            V = Q[: self.x.shape[0], : self.x.shape[0]] + Q[: self.x.shape[0], self.x.shape[0] :] @ self.K[t] + self.K[t].T @ Q[self.x.shape[0] :, : self.x.shape[0]] + self.K[t].T @ Q[self.x.shape[0] :, self.x.shape[0] :] @ self.K[t]
            v = q[: self.x.shape[0]] + Q[: self.x.shape[0], self.x.shape[0] :] @ self.k[t] + self.K[t].T @ q[self.x.shape[0] :] + self.K[t].T @ Q[self.x.shape[0] :, self.x.shape[0] :] @ self.k[t]

        ## Forward Recursion ##
        for t in range(self.T):
            self.u[t] = self.K[t] @ self.x + self.k[t]
            self.x = self.F[t] @ np.vstack((self.x, self.u[t])) + self.f[t]

        return

    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(LQR_help)

    def __str__(self):
        return "<LQR Model>"


# string to print when calling help() method
LQR_help = """

-------------------- *** --------------------

Id: LQR

Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.

Methods:

    initialize(F, f, C, c, T, x)
        Description:
            Initialize the dynamics of the model
        Args:
            F (float/numpy.ndarray): past value contribution coefficients
            f (float/numpy.ndarray): bias coefficients
            C (float/numpy.ndarray): quadratic cost coefficients
            c (float/numpy.ndarray): linear cost coefficients
            T (postive int): number of timesteps
            x (float/numpy.ndarray): initial state

    step()
        Description: Updates internal parameters and then returns the
        	estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions

    predict()
        Description:
            Returns estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions

    update()
        Description:
        	Updates internal parameters
        Args:
            None

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""