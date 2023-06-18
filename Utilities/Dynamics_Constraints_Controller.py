import numpy as np

rx = 2
ru = 1
ry = 2
A = np.array([[0.95, 0.1],[0, 0.90]])
B = np.array([[0, 1]]).T
C = np.array([[1, 0], [0, 1]])
Q = 0.2 * np.diag(np.ones(2,))
R = 0.4 * np.diag(np.ones(2,))

def stateDynamics(x, u, w):
    x = x.squeeze()
    w = w.squeeze()
    u = np.atleast_2d(u.squeeze()).T
    x_p = A @ x + (B @ u).squeeze() + w
    return np.atleast_2d(x_p.squeeze()).T

def measurementDynamics(x, u):
    x = x.squeeze()
    u = np.atleast_2d(u.squeeze()).T
    y = C @ x
    return np.atleast_2d(y.squeeze()).T

def controller(x): #Here you define your controller, whether an MPC, SMPC, CBF, PID, whatever...
    u=-0.05*x[0]*x[1]
    return u

def Constraints(u,state,Ulim):
    ConstraintX=0
    ConstraintU=0
    x=state[0]
    y=state[1]
    if ((3<x<5)&(-4<y<2)|(-2<x<5)&(-7<y<-4)):
            ConstraintX=1
    if (abs(u)>Ulim):
            ConstraintU=1
    return np.array([ConstraintX, ConstraintU])

