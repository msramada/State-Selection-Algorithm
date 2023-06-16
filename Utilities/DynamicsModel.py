import numpy as np


A = np.array([[0.95, 0.1],[0, 0.90]])
B = np.array([[0, 1]]).T
C = np.array([[1, 0], [0, 1]])
Q = 0.2 * np.diag(np.ones(2,))
R = 0.4 * np.diag(np.ones(2,))

def stateDynamics(x, u):
    x = x.squeeze()
    u = np.atleast_2d(u.squeeze()).T
    x_p = A @ x + (B @ u).squeeze()
    return x_p.squeeze()

def measurementDynamics(x, u):
    x = x.squeeze()
    u = np.atleast_2d(u.squeeze()).T
    y = C @ x
    return y.squeeze()

