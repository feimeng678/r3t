#!/usr/bin/env python3

import numpy as np
from koopman_operator import KoopmanOperator
from task import Task, Adjoint
import scipy.linalg
from group_theory import VecTose3, TransToRp, RpToTrans
from lqr import FiniteHorizonLQR
from quatmath import euler2mat
import math

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from polytope_symbolic_system.examples.pendulum import Pendulum



np.set_printoptions(precision=4, suppress=True)
#np.random.seed(50) ### set the seed for reproducibility

def alcontrol_step(starting_state=None, target_state=None, states_list=None, sz=None, ndsz=None):
    pend = Pendulum()
    # ntc = Pendulum.set_ntc()
    ### the timing parameters for the quad are used
    ### to construct the koopman operator and the adjoint dif-eq
    koopman_operator = KoopmanOperator(pend.time_step)
    adjoint = Adjoint(pend.time_step)
    simulation_time = 1000
    horizon = 20  ### time horizon
    # control_reg = np.diag([1.] * 4) ### control regularization
    control_reg = np.diag([1.]) * 25
    inv_control_reg = np.linalg.inv(control_reg)  ### precompute this
    # default_action = lambda x : np.random.uniform(-0.1, 0.1, size=(4,)) ### in case lqr control returns NAN
    default_action = lambda x: np.random.uniform(-0.1, 0.1, size=(1,))
    ### initialize the state
    state = starting_state#np.array([-0.5 * math.pi, 0])
    # target_position = np.array([-1.5 * math.pi, 0.])
    task = Task()  ### this creates the task
    err = np.zeros([simulation_time, 2])
    # posi = np.zeros([simulation_time, 3])
    al_states_list = states_list
    for t in range(int(sz / ndsz)):
        #### measure state and transform through koopman observables
        m_state = get_koop_measurement(starting_state)  # z
        t_state = koopman_operator.transform_all_state(m_state)
        #   err[t] = np.linalg.norm(m_state[:3] - target_orientation) + (m_state[3:]))
        err[t, :] = np.array([state - target_state])  # + np.linalg.norm(m_state[3:6] - target_orientation) + np.linalg.norm(m_state[6:])task.target_expanded_state
        Kx, Ku = koopman_operator.get_linearization()  ### 21*21, 21*4 grab the linear matrices
        lqr_policy = FiniteHorizonLQR(Kx, Ku, task.Q, task.R, task.Qf, horizon=horizon)  # instantiate a lqr controller
        lqr_policy.set_target_state(task.cal_target_expanded_state(target_state))  ## set target state to koopman observable state, task.target_expanded_state
        lqr_policy.sat_val = pend.sat_val  ### set the saturation value
        ### forward sim the koopman dynamical system.
        # traj is z=20*(21*1) fdx 20*(21*21) fdu20(21*4)
        # (optimal policy) action_schedule=policy(state): -K[0]*err
        trajectory, fdx, fdu, action_schedule = koopman_operator.simulate(t_state, horizon, policy=lqr_policy)  #(here fdx, fdu is just Kx, Ku in a list)
        ldx, ldu = task.get_linearization_from_trajectory(trajectory, action_schedule)  # ldx 19*(21) ldu19*(4*1)
        mudx = lqr_policy.get_linearization_from_trajectory(trajectory)  # control_gains 20*(4*21)

        rhof = task.mdx(trajectory[-1])  ### get terminal condition for adjoint
        rho = adjoint.simulate_adjoint(rhof, ldx, ldu, fdx, fdu, mudx, horizon)

        ustar = -np.dot(inv_control_reg, fdu[0].T.dot(rho[0])) + lqr_policy(t_state)
        ustar = np.clip(ustar, -pend.sat_val, pend.sat_val)  ### saturate control

        if np.isnan(ustar).any():
            ustar = default_action(None)

        ### advacne quad subject to ustar
        next_state = pend.step(state, ustar)
        # next_position=get_position(next_state)

        ### update the koopman operator from data
        koopman_operator.compute_operator_from_data(get_koop_measurement(state),
                                                    ustar,
                                                    get_koop_measurement(next_state))

        state = next_state  ### backend : update the simulator state
        al_states_list.append(state)
        # position=next_position
        ### we can also use a decaying weight on inf gain
        task.inf_weight = 100.0 * (0.99 ** (t))
        # if t % 100 == 0:
        #     print('time : {}, pose : {}, {}'.format(t*quad.time_step,
        #                                             get_all_measurement(state), ustar))
    t = [i * pend.time_step for i in range(simulation_time)]
    return state, al_states_list

def get_koop_measurement(x):
    return np.array(x)

def get_all_measurement(x): ##eq29 9*1
    g = x[0:16].reshape((4,4)) ## SE(3) matrix
    R,p = TransToRp(g)
    twist = x[16:]##select start from 17th
    grot = np.dot(R, [0., 0., -9.81]) ## gravity vec related to body frame
    return np.concatenate((p, grot, twist))

def get_position(x):
    g = x[0:16].reshape((4,4))
    R,p = TransToRp(g)
    return p

def main():

    # quad = Quad() ### instantiate a quadcopter
    pend = Pend()
    ntc = Pend.set_ntc()

    ### the timing parameters for the quad are used
    ### to construct the koopman operator and the adjoint dif-eq
    koopman_operator = KoopmanOperator(pend.time_step)
    adjoint = Adjoint(pend.time_step)

    # task = Task()
    simulation_time = 1500
    horizon = 20 ### time horizon
    sat_val = 1.0 ### saturation value
    # control_reg = np.diag([1.] * 4) ### control regularization
    control_reg = np.diag([1.])*25

    inv_control_reg = np.linalg.inv(control_reg) ### precompute this
    # default_action = lambda x : np.random.uniform(-0.1, 0.1, size=(4,)) ### in case lqr control returns NAN
    default_action = lambda x: np.random.uniform(-0.1, 0.1, size=(1,))

    ### initialize the state
    state = np.array([-0.5*math.pi, 0])
    target_position=np.array([-1.5*math.pi, 0.])


    task = Task(target_position) ### this creates the task

    err = np.zeros([simulation_time,2])
    posi=np.zeros([simulation_time,3])
    for t in range(simulation_time):

        #### measure state and transform through koopman observables
        m_state = get_koop_measurement(state)   #z

        t_state = koopman_operator.transform_all_state(m_state)
     #   err[t] = np.linalg.norm(m_state[:3] - target_orientation) + (m_state[3:]))
        err[t,:] = np.array([state-target_position])# + np.linalg.norm(m_state[3:6] - target_orientation) + np.linalg.norm(m_state[6:])
        Kx, Ku = koopman_operator.get_linearization() ### 21*21, 21*4 grab the linear matrices
        lqr_policy = FiniteHorizonLQR(Kx, Ku, task.Q, task.R, task.Qf, horizon=horizon) # instantiate a lqr controller
        lqr_policy.set_target_state(task.target_expanded_state) ## set target state to koopman observable state
        lqr_policy.sat_val = sat_val ### set the saturation value

        ### forward sim the koopman dynamical system.
        # traj is z=20*(21*1) fdx 20*(21*21) fdu20(21*4)
        #(optimal policy) action_schedule=policy(state): -K[0]*err
        trajectory, fdx, fdu, action_schedule = koopman_operator.simulate(t_state, horizon, policy=lqr_policy)  #fixme#(here fdx, fdu is just Kx, Ku in a list)

        ldx, ldu = task.get_linearization_from_trajectory(trajectory, action_schedule) #ldx 19*(21) ldu19*(4*1)
        mudx = lqr_policy.get_linearization_from_trajectory(trajectory) #control_gains 20*(4*21)

        rhof = task.mdx(trajectory[-1]) ### get terminal condition for adjoint
        rho = adjoint.simulate_adjoint(rhof, ldx, ldu, fdx, fdu, mudx, horizon)

        ustar = -np.dot(inv_control_reg, fdu[0].T.dot(rho[0])) + lqr_policy(t_state)
        ustar = np.clip(ustar, -sat_val, sat_val) ### saturate control

        if np.isnan(ustar).any():
            ustar = default_action(None)

        ### advacne quad subject to ustar
        next_state = pend.step(state, ustar)
        # next_position=get_position(next_state)

        ### update the koopman operator from data
        koopman_operator.compute_operator_from_data(get_koop_measurement(state),
                                                    ustar,
                                                    get_koop_measurement(next_state))

        state = next_state ### backend : update the simulator state
        # position=next_position
        ### we can also use a decaying weight on inf gain
        task.inf_weight = 100.0 * (0.99**(t))
        # if t % 100 == 0:
        #     print('time : {}, pose : {}, {}'.format(t*quad.time_step,
        #                                             get_all_measurement(state), ustar))

    t = [i * pend.time_step for i in range(simulation_time)]

    plt.figure(1)
    plt.plot(t, err[:, 0])
    plt.plot(t, err[:, 1])
    plt.xlabel('t')
    plt.ylabel('tracking error')
    plt.show()

    # plt.plot(t, posi)
    # plt.xlabel('t')
    # plt.ylabel('position')
    # plt.show()

    # plt.figure(2)
    # ax1 = plt.axes(projection='3d')
    # x=posi[:,0]
    # y = posi[:,1]
    # z = posi[:,2]
    # ax1.plot3D(x,y,z, 'gray')  # 绘制空间曲线
    # plt.show()


if __name__=='__main__':
    main()
