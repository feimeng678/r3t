import autograd.numpy as np
import matplotlib.pyplot as plt
from koopman_operator import psix, psiu, NUM_OBS_, NUM_STATE_OBS_, NUM_ACTION_OBS_, psix_all


class Adjoint(object):

    def __init__(self, sampling_time):
        self.sampling_time = sampling_time

    def rhodt(self, rho, ldx, ldu, fdx, fdu, mudx):
        return - np.dot((fdx + np.dot(fdu, mudx)).T, rho) - (ldx + np.dot(mudx.T, ldu))

    def simulate_adjoint(self, rhof, ldx, ldu, fdx, fdu, mudx, N):
        rho = [None] * N
        rho[N-1] = rhof
        for i in reversed(range(1, N)):
            rhodt = self.rhodt(rho[i], ldx[i-1], ldu[i-1], fdx[i-1], fdu[i-1], mudx[i-1])
            rho[i-1] = rho[i] - rhodt * self.sampling_time
        return rho


class Task(object):

    
    def __init__(self):
        Qdiag = np.zeros(NUM_STATE_OBS_)
        Qdiag[0] = 1 # th bigger righter
        Qdiag[1] = 0.25  # dth bigger lefter
        Qdiag[2:4] = 0.1 # sin cos th bigger
        #Qdiag[4] = 0.0001 # omega
        # Qdiag[9:12] = 5.  # v
        self.Q = np.diag(Qdiag)*10.0

        Qfdiag = np.ones(NUM_STATE_OBS_)
        Qfdiag[2:] = 10.0
        self.Qf = 0*np.diag(Qfdiag)
        #self.Qf *= 0
        # self.target_state = np.zeros(9)
        # self.target_state[2] = -9.81
        # self.target_expanded_state = psix(self.target_state)
        # self.R = np.diag([1.0]*4)
        self.R = np.diag([1.0])*0.01 #0.1
        self.inf_weight = 100.0
        self.eps = 1e-5
        self.final_cost = 0
        self.sys_sta_num = 2
        self.target_expanded_state = np.zeros(NUM_STATE_OBS_)

        # self.target_all_state = np.zeros(target_state.shape)
        # n = target_state.shape[0]
        # self.target_all_state[:n] = target_state
        # self.target_all_state[3:6] = target_orientation
        #self.target_expanded_state = psix_all(self.target_all_state)

    def cal_target_expanded_state(self, target_state):
        self.target_all_state = np.zeros(self.sys_sta_num)
        n = target_state.shape[0]
        self.target_all_state[:n] = target_state
        self.target_expanded_state = psix_all(self.target_all_state)
        return self.target_expanded_state

    def l(self, state, action):
        error = state - self.target_expanded_state
        error_q = np.dot(self.Q, error)
        action_r = np.dot(self.R, action)
        return np.dot(error, error_q) + np.dot(action, action_r) + self.inf_weight / (np.dot(state, state)+self.eps)

    def get_stab_cost(self, state):
        # return np.dot(state[3:9], state[3:9])
        return np.dot(state[0:2], state[0:2])

    def information_gain(self, state):
        return np.dot(state, state)

    def ldx(self, zstate, action):
        error = zstate - self.target_expanded_state
        d_err = np.zeros(zstate.shape)
        d_err = zstate
        return np.dot(self.Q, error) - self.inf_weight * 2.0 * d_err / (np.dot(zstate, zstate) + self.eps) ** 2

    def ldu(self, state, action):
        action_r = np.dot(self.R, action)
        return action_r

    def m(self, state):
        error = state - self.target_expanded_state
        error_q = np.dot(self.Qf, error)
        return np.dot(error, error_q)*self.final_cost 

    def mdx(self, state):
        error = state - self.target_expanded_state
        return np.dot(self.Qf, error) * self.final_cost

    def get_linearization_from_trajectory(self, trajectory, actions): #traj(t_state)
        return [self.ldx(state, action) for state, action in zip(trajectory, actions)], [self.ldu(state, action) for state, action in zip(trajectory, actions)]

    def trajectory_cost(self, trajectory, actions):
        total_cost = 0.0
        for state, action in zip(trajectory, actions):
            total_cost += self.l(state, action)

        return total_cost + self.m(trajectory[-1])

