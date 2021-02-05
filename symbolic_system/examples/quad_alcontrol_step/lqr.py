import numpy as np
from math import sin, cos
from autograd import jacobian
from koopman_operator import NUM_STATE_OBS_
from quad import Quad


class FiniteHorizonLQR(object):
    def __init__(self, A, B, Q, R, F, horizon=10):  #Kx, Ku, task.Q, task.R, task.Qf
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Rinv = np.linalg.inv(R)
        self.F = F
        self.time_step = 1/200
        self.horizon = horizon
        self.sat_val = 10.0
        self.target_state = None
        self.active = 1
        self.final_cost = 0
        self.SN = np.int(self.horizon / self.time_step)

    def set_target_state(self, target):
        self.target_state = target

    def get_control_gains(self):
        P = [None] * self.horizon
        P[-1] = self.F.copy()
        K = [None] * (self.horizon)
        r = [None] * (self.horizon)
        K[-1] = self.Rinv.dot(self.B.T.dot(P[-1]))
        r[-1] = self.F.dot(self.target_state*0.0)
        for i in reversed(range(1, self.horizon)):
            PB = np.dot(P[i], self.B)
            BP = np.dot(self.B.T, P[i])
            PBRB = np.dot(PB, np.dot(self.Rinv, self.B.T))
            Pdot = - (np.dot(self.A.T, P[i]) + np.dot(P[i], self.A) - np.dot(np.dot(PB, self.Rinv), BP) + self.Q)
            rdot = -(self.A.T.dot(r[i]) - self.Q.dot(self.target_state) - PBRB.dot(r[i]))
            P[i-1] = P[i] - Pdot*self.time_step
            K[i-1] = self.Rinv.dot(self.B.T.dot(P[i]))
            r[i-1] = r[i] - rdot*self.time_step
        return K, r

    def ntc_get_control_gains(self, x):
        N = np.zeros([4, 1])
        P = np.zeros([4, 1])
        M = np.zeros([1, 1])
        D = np.zeros([4, 1])
        G = 0
        for i in range(self.SN):
            psi_jaco, psi_f_bar = self.psi_jaco_fun(x)
            # self.A
            S0 = np.zeros([4, 4])
            V0 = np.zeros([2, 2])
            P0 = psi_jaco
            S1 = np.zeros([4, 1])
            V1 = -psi_f_bar
            P1 = 0
            for j in range(i in range (self.SN)):
                S0 = S0 + self.time_step * (S0.T*self.A+self.A.T * S0-(S0.T*self.B+N) * self.Rinv * (S0*self.B+N).T+self.Q)
                P0 = P0 + self.time_step * P0 * (self.A - self.B * self.Rinv * (S0.T*self.B+N).T)
                V0 = V0 - self.time_step * (P0 * self.B * self.Rinv * self.B.T*P0.T)
                S1 = S1 + self.time_step * ((self.A.T-(S0.T * self.B+N) * self.Rinv * self.B.T)*S1-(S0.T * self.B * self.Rinv * M - S0.T*D-P))
                V1 = V1 + self.time_step * P0 * (D - self.B * self.Rinv * (M + self.B.T*S1))
                P1 = P1 + self.time_step * (-0.5 * (M+self.B.T*S1).T * self.Rinv * (M + self.B.T*S1)+S1.T * D+G)

            v=-np.linalg.pinv(V0) * (P0 * x + V1)
            u = -self.Rinv * (self.B.T*(S0*x+P0.T * v+S1) + N.T*x+M)
        return K, r

    def psi_jaco_fun(self, x):
        psi_f, psi_x = Pend.set_ntc(x)
        psidx1 = jacobian(self.psi[0], argnum=0)
        psidx2 = jacobian(self.psi[1], argnum=1)
        psi_jaco = np.concatenate(psidx1, psidx2, axis=1)
        psi_f_bar = psi_f + psi_jaco * x - psi_x
        return psi_jaco, psi_f_bar


    def __call__(self, state):
        K,r = self.get_control_gains()
        ref = -self.Rinv.dot(self.B.T).dot(r[0])
        return np.clip(-K[0].dot(state-self.target_state), -self.sat_val, self.sat_val)
    
    def get_linearization_from_trajectory(self, trajectory):
        K,_ = self.get_control_gains()
        return [-k for k in K]
