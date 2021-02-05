from koopman_system import notebookfns as n
import numpy as np
import pickle
from autograd import jacobian
from autograd.numpy import sin, cos


class Pend(object):
    num_states = 2
    num_actions = 1
    time_step = 1.0/100.0
    inertia_matrix = np.ones(2) #np.diag([0.04, 0.0375, 0.0675])
    # inv_inertia_matrix = np.linalg.inv(inertia_matrix)

    def __init__(self):
        self.fdx = jacobian(self.f, argnum=0)
        self.fdu = jacobian(self.f, argnum=1)
        l = 1
        self.m = 1
        self.g = 9.8
        inv_mass = 1.0 / self.m
        self.b = 0.1

# fname = './koopman_system/Pendulum_2020_09_29_02_43_28_836240_model.pkl'
#             with open(fname, 'rb') as f:
#                 params = pickle.load(f, encoding='latin1')
#             W, b = n.load_weights_koopman(fname, len(params['widths']) - 1, len(params['widths_omega_real']) - 1,
#                                       params['num_real'], params['num_complex_pairs'])
#             x_env = extract_variable_value_from_env(self.x, env)
#             u_env = extract_variable_value_from_env(self.u, env)
#             yk, ykplus1, ykplus2, ykplus3, xk_recon, xkplus1, xkplus2, xkplus3 = n.ApplyKoopmanNetOmegas(x_env, W, b,\
#                 params['delta_t'], params['num_real'], params['num_complex_pairs'], params['num_encoder_weights'], \
#                 params['num_omega_weights'], params['num_decoder_weights'])


    def f(self, x, uu):
        u = np.clip(uu, -10000, 10000)
        dtheta=x[1]
        ddtheta=-self.b*x[1]+self.m*self.g*sin(x[0])+u[0]
        return np.array([np.float(dtheta), ddtheta])
        # g = x[0:16].reshape((4, 4))
        # R, p = TransToRp(g)
        #
        # omega = x[16:19]
        # v = x[19:]
        # twist = x[16:]
        #
        # # u[0] *= 0.8
        # F = self.kt * (u[0] + u[1] + u[2] + u[3])
        # M = np.array([
        #     self.kt * self.arm_length * (u[1] - u[3]),
        #     self.kt * self.arm_length * (u[2] - u[0]),
        #     self.km * (u[0] - u[1] + u[2] - u[3])
        # ])
        #
        # inertia_dot_omega = np.dot(self.inertia_matrix, omega)
        # inner_rot = M + cross(inertia_dot_omega, omega)
        #
        # omegadot = np.dot(
        #     self.inv_inertia_matrix,
        #     inner_rot
        # ) - self.ang_damping * omega
        #
        # vdot = self.inv_mass * F * np.array([0., 0., 1.]) - cross(omega, v) - 9.81 * np.dot(R.T, np.array(
        #     [0., 0., 1.])) - self.vel_damping * v
        #
        # dg = np.dot(g, VecTose3(twist)).ravel()

    def step(self, state, action):
        k1 = self.f(state, action) * self.time_step
        k2 = self.f(state + k1 / 2.0, action) * self.time_step
        k3 = self.f(state + k2 / 2.0, action) * self.time_step
        k4 = self.f(state + k3, action) * self.time_step
        return state + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
