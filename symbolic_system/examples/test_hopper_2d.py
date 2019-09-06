import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from polytope_symbolic_system.examples.hopper_2d import Hopper_2d
from rg_rrt_star.symbolic_system.symbolic_system_rg_rrt_star import SymbolicSystem_RGRRTStar, PolytopeReachableSet
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from pypolycontain.lib.polytope import polytope
from pypolycontain.lib.operations import distance_point_polytope, distance_polytopes
from rg_rrt_star.utils.visualization import visualize_node_tree_2D
from collections import deque
import time
from datetime import datetime
import os

class Hopper2D_ReachableSet(PolytopeReachableSet):
    def __init__(self, parent_state, polytope_list, epsilon=1e-3, contains_goal_function = None, deterministic_next_state = None, ground_height_function = lambda x:0):
        PolytopeReachableSet.__init__(self, parent_state, polytope_list, epsilon, contains_goal_function, deterministic_next_state)
        self.ground_height_function = ground_height_function
    def plan_collision_free_path_in_set(self, goal_state, return_deterministic_next_state = False):
        #fixme: support collision checking
        #check for impossible configurations

        # tipped over
        if goal_state[2]>np.pi/2 or goal_state[2]<-np.pi/2:
            print('leg tipped over')
            if return_deterministic_next_state:
                return np.inf, None, None
            else:
                return np.inf, None

        # body attitude is off
        if goal_state[3]>np.pi/2 or goal_state[3]<-np.pi/2:
            print('body attitude off')
            if return_deterministic_next_state:
                return np.inf, None, None
            else:
                return np.inf, None

        # stuck in the ground
        if goal_state[1]<self.ground_height_function(goal_state[0])-.8:
            print('stuck in ground')
            if return_deterministic_next_state:
                return np.inf, None, None
            else:
                return np.inf, None

        # stuck in the ground
        if goal_state[4]<0.2:
            print('r invalid')
            if return_deterministic_next_state:
                return np.inf, None, None
            else:
                return np.inf, None
        #
        # is_contain, closest_state = self.contains(goal_state)
        # if not is_contain:
        #     print('Warning: this should never happen')
        #     return np.linalg.norm(self.parent_state-closest_state), deque([self.parent_state, closest_state]) #FIXME: distance function

        # Simulate forward dynamics if there can only be one state in the next timestep
        if not return_deterministic_next_state:
            return np.linalg.norm(self.parent_state-goal_state), deque([self.parent_state, goal_state])
        return np.linalg.norm(self.parent_state-goal_state), deque([self.parent_state, goal_state]), self.deterministic_next_state

class Hopper2D_RGRRTStar(SymbolicSystem_RGRRTStar):
    def __init__(self, sys, sampler, step_size, contains_goal_function = None):
        self.sys = sys
        self.step_size = step_size
        self.contains_goal_function = contains_goal_function
        def compute_reachable_set(state):
            '''
            Compute zonotopic reachable set using the system
            :param h:
            :return:
            '''
            deterministic_next_state = None
            reachable_set_polytope = self.sys.get_reachable_polytopes(state, step_size=self.step_size)
            # if state[1] <= 0:
            #     print('state', state)
            #     print("T", reachable_set_polytope[0].T)
            #     print("t", reachable_set_polytope[0].t)
            #     print("H", reachable_set_polytope[0].P.H)
            #     print("h", reachable_set_polytope[0].P.h)
            #TODO: collision check here
            if np.all(self.sys.get_linearization(state=state).B == 0):
                deterministic_next_state = self.sys.forward_step(starting_state=state, modify_system=False, return_as_env=False, step_size=self.step_size)
            return Hopper2D_ReachableSet(state,reachable_set_polytope, contains_goal_function=self.contains_goal_function, deterministic_next_state=deterministic_next_state)
        SymbolicSystem_RGRRTStar.__init__(self, sys, sampler, step_size, contains_goal_function, compute_reachable_set)

def test_hopper_2d_planning():
    initial_state = np.asarray([0.5, 1, 0, 0, 5, 0, 0., 0., 0., 0.])
    hopper_system = Hopper_2d(initial_state=initial_state)
    # [theta1, theta2, x0, y0, w]
    # from x0 = 0 move to x0 = 5
    goal_state = np.asarray([5.,0.,0.,0.,5.,0.,0.,0.,0.,0.])
    goal_tolerance = [0.1,10,10,10,10,5,5,5,5,5]
    step_size = 1e-1
    #TODO
    def uniform_sampler():
        rnd = np.random.rand(10)
        rnd[0] = (rnd[0]-.2)*14
        rnd[1] = (rnd[1]-0.4)*10
        rnd[2] = (rnd[2] - 0.5) * 2 * np.pi/3
        rnd[3] = (rnd[3]-0.5) * 2 * np.pi/4
        rnd[4] = (rnd[4]-0.5)*2*5+5
        rnd[5] = (rnd[5]-0.2)*2*10
        rnd[6] = (rnd[6] - 0.5) * 2 * 10
        rnd[7] = (rnd[7] - 0.5) * 2 * 4
        rnd[8] = (rnd[8] - 0.5) * 2 * 4
        rnd[9] = (rnd[9] - 0.1) * 2 * 20
        # goal_bias = np.random.rand(1)
        return rnd

    # def gaussian_mixture_sampler():
    #     gaussian_ratio = 0.4
    #     rnd = np.random.rand(2)
    #     rnd[0] = np.random.normal(l+0.5*p,2*p)
    #     rnd[1] = (np.random.rand(1)-0.5)*2*4
    #     if np.random.rand(1) > gaussian_ratio:
    #         return uniform_sampler()
    #     return rnd
    #
    # def action_space_mixture_sampler():
    #     action_space_ratio = 0.08
    #     if np.random.rand(1) < action_space_ratio:
    #         rnd = np.random.rand(2)
    #         rnd[0] = rnd[0]*p*1.2+l
    #         rnd[1] = (rnd[1]-0.5)*2*8
    #         return rnd
    #     else:
    #         rnd = np.random.rand(2)
    #         rnd[0] = rnd[0]*4 + l
    #         rnd[1] = (rnd[1] - 0.5) * 2 * 2
    #         goal_bias = np.random.rand(1)
    #         if goal_bias < 0.35:
    #             rnd[0] = np.random.normal(goal_state[0],1.5)
    #             rnd[1] = np.random.normal(goal_state[1],1.5)
    #             return rnd
    #         return rnd

    def contains_goal_function(reachable_set, goal_state):
        distance = np.inf
        projection = None
        for p in reachable_set.polytope_list:
            d, proj = distance_point_polytope(p, goal_state)
            if d<distance:
                projection = np.asarray(proj)
                distance = d
        if np.all(abs(projection-goal_state)<goal_tolerance):
            return True
        return False

    rrt = Hopper2D_RGRRTStar(hopper_system, uniform_sampler, step_size, contains_goal_function=contains_goal_function)
    found_goal = False
    experiment_name = datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H-%M-%S')

    duration = 0
    os.makedirs('RRT_Hopper_2d_'+experiment_name)
    max_iterations = 10000
    for itr in range(max_iterations):
        start_time = time.time()
        if rrt.build_tree_to_goal_state(goal_state, stop_on_first_reach=True, allocated_time= 15, rewire=True, explore_deterministic_next_state=True) is not None:
            found_goal = True
        end_time = time.time()
        #get rrt polytopes
        polytope_reachable_sets = rrt.reachable_set_tree.id_to_reachable_sets.values()
        reachable_polytopes = []
        explored_states = []
        for prs in polytope_reachable_sets:
            reachable_polytopes.extend(prs.polytope_list)
            explored_states.append(prs.parent_state)
        # print(explored_states)
        # print(len(explored_states))
        # print('number of nodes',rrt.node_tally)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig, ax = visualize_node_tree_2D(rrt, fig, ax, s=0.5, linewidths=0.15, show_path_to_goal=found_goal, dims=[0,1])
        # for explored_state in explored_states:
        #     plt.scatter(explored_state[0], explored_state[1], facecolor='red', s=6)
        ax.scatter(initial_state[0], initial_state[1], facecolor='red', s=5)
        # ax.set_aspect('equal')
        # plt.plot([l+p, l+p], [-2.5, 2.5], 'm--', lw=1.5)
        plt.plot([5,5], [-2.5, 2.5], 'g--', lw=1.5)

        # ax.set_xlim(left=0)
        plt.xlabel('$x_0$')
        plt.ylabel('$y_0$')
        duration += (end_time-start_time)
        plt.title('RRT Tree after %.2f seconds (explored %d nodes)' %(duration, len(polytope_reachable_sets)))
        plt.savefig('RRT_Hopper_2d_'+experiment_name+'/%.2f_seconds.png' % duration, dpi=500)

        # plt.show()
        plt.clf()
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig, ax = visualize_node_tree_2D(rrt, fig, ax, s=0.5, linewidths=0.15, show_path_to_goal=found_goal, dims=[1,4])
        # for explored_state in explored_states:
        #     plt.scatter(explored_state[0], explored_state[1], facecolor='red', s=6)
        ax.scatter(initial_state[1], initial_state[4], facecolor='red', s=5)

        # ax.set_xlim(left=0)
        plt.xlabel('$y_0$')
        plt.ylabel('$r$')
        plt.title('RRT Tree after %.2f seconds (explored %d nodes)' %(duration, len(polytope_reachable_sets)))
        plt.savefig('RRT_Hopper_2d_'+experiment_name+'/%.2f_seconds_2.png' % duration, dpi=500)

        # plt.show()
        plt.clf()
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig, ax = visualize_node_tree_2D(rrt, fig, ax, s=0.5, linewidths=0.15, show_path_to_goal=found_goal, dims=[0,4])
        # for explored_state in explored_states:
        #     plt.scatter(explored_state[0], explored_state[1], facecolor='red', s=6)
        ax.scatter(initial_state[0], initial_state[4], facecolor='red', s=5)

        # ax.set_xlim(left=0)
        plt.xlabel('$x_0$')
        plt.ylabel('$r$')
        plt.title('RRT Tree after %.2f seconds (explored %d nodes)' %(duration, len(polytope_reachable_sets)))
        plt.savefig('RRT_Hopper_2d_'+experiment_name+'/%.2f_seconds_3.png' % duration, dpi=500)


        # plt.show()
        plt.clf()
        plt.close()
        if found_goal:
            break


if __name__=='__main__':
    test_hopper_2d_planning()