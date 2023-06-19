import numpy as np
import control 

rx = 2
ru = 1
ry = 2
A = np.array([[0.95, 0.1],[0, 0.90]])
B = np.array([[0, 1]]).T
C = np.array([[1, 0], [0, 1]])
Q = 0.2 * np.diag(np.ones(2,))
R = 0.4 * np.diag(np.ones(2,))
# LQR design params
Q_lqr = np.diag(1.0 * np.ones(rx,))
R_lqr = np.diag(1.0 * np.ones(ru,))
Klqr, _, _ = control.dlqr(A, B, Q_lqr, R_lqr)
Klqr = - Klqr

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

def Controller(x): #Here you define your controller, whether an MPC, SMPC, CBF, PID, whatever...
    u = Klqr @ x
    return u


H = np.array([[1, 0], [0, 1]])
b = np.array([[5,2]]).T
def CostAndConstraints(Control_seq,xk2prime):
        cost = (xk2prime ** 2).sum() + (Control_seq ** 2).sum()
        # Bounds on the control
        Control_violations = abs(Control_seq) > 5
        Control_violations = np.append(Control_violations, False)
        # Linear state constraints violation: Hx > b
        State_violations = (H @ xk2prime) > b
        State_violations = State_violations.sum(axis=0)>0
        number_of_violations = (State_violations.squeeze() | Control_violations.squeeze()).sum()
        return cost, number_of_violations

