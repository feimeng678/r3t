import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from polytope_symbolic_system.examples.pendulum import Pendulum
from rg_rrt_star.symbolic_system.symbolic_system_basic_rrt import SymbolicSystem_RGRRT
from pypolycontain.visualization.visualize_2D import visualize_2D_AH_polytope
from pypolycontain.lib.operations import distance_point_polytope
from rg_rrt_star.utils.visualization import visualize_node_tree_2D
import time
from datetime import datetime
import os

def test_pendulum_planning():
    initial_state = np.zeros(2)
    pendulum_system = Pendulum(initial_state= initial_state, input_limits=np.asarray([[-0.1],[0.1]]), m=1, l=0.5, g=9.8, b=0.1)
    goal_state = np.asarray([np.pi,0.0])
    goal_state_2 = np.asarray([-np.pi,0.0])
    step_size = 0.075
    def uniform_sampler():
        rnd = np.random.rand(2)
        rnd[0] = (rnd[0]-0.5)*2*1.5*np.pi
        rnd[1] = (rnd[1]-0.5)*2*9
        goal_bias_rnd = np.random.rand(1)
        # if goal_bias_rnd <0.2:
        #     return goal_state + [2*np.pi*np.random.randint(-1,1),0] + [np.random.normal(0,0.8),np.random.normal(0,1.5)]
        return rnd

    def gaussian_mixture_sampler():
        gaussian_ratio = 0.0
        rnd = np.random.rand(2)
        rnd[0] = np.random.normal(goal_state[0],1)
        rnd[1] = np.random.normal(goal_state[1],1)
        if np.random.rand(1) > gaussian_ratio:
            return uniform_sampler()
        return rnd

    def ring_sampler():
        theta = np.random.rand(1)*2*np.pi
        rnd = np.zeros(2)
        r = np.random.rand(1)+2.5
        rnd[0] = r*np.cos(theta)
        rnd[1] = r*np.sin(theta)
        return rnd

    def big_gaussian_sampler():
        rnd = np.random.rand(2)
        rnd[0] = np.random.normal(0,1.5)
        rnd[1] = np.random.normal(0,3)
        goal_bias_rnd = np.random.rand(1)
        if goal_bias_rnd <0.05:
            return goal_state
        elif goal_bias_rnd < 0.1:
            return np.asarray([-np.pi,0.0])
        return rnd

    def reached_goal_function(state, goal_state):
        if np.linalg.norm(state-goal_state)<1.5e-1 or np.linalg.norm(state-goal_state_2)<1.5e-1:
            return True
        return False

    rrt = SymbolicSystem_RGRRT(pendulum_system, uniform_sampler, step_size, reached_goal_function=reached_goal_function)
    found_goal = False
    experiment_name = datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H-%M-%S')

    duration = 0
    os.makedirs('RG_RRT_Pendulum_'+experiment_name)
    while(1):
        start_time = time.time()
        if rrt.build_tree_to_goal_state(goal_state,stop_on_first_reach=True, allocated_time= 100, rewire=False, explore_deterministic_next_state=False) is not None:
            found_goal = True
        end_time = time.time()

        if found_goal:
            p = rrt.goal_node.parent.state
            if np.linalg.norm(p-np.asarray([np.pi,0.0])) < np.linalg.norm(p-np.asarray([-np.pi,0.0])):
                goal_override = np.asarray([np.pi,0.0])
            else:
                goal_override = np.asarray([-np.pi, 0.0])
        # Plot state tree
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if found_goal:
            fig, ax = visualize_node_tree_2D(rrt, fig, ax, s=0.5, linewidths=0.15, show_path_to_goal=found_goal, goal_override=goal_override)
        else:
            fig, ax = visualize_node_tree_2D(rrt, fig, ax, s=0.5, linewidths=0.15, show_path_to_goal=found_goal)
        # fig, ax = visZ(reachable_polytopes, title="", alpha=0.07, fig=fig,  ax=ax, color='gray')
        # for explored_state in explored_states:
        #     plt.scatter(explored_state[0], explored_state[1], facecolor='red', s=6)
        ax.scatter(initial_state[0], initial_state[1], facecolor='red', s=5)
        ax.scatter(goal_state[0], goal_state[1], facecolor='green', s=5)
        ax.scatter(goal_state[0]-2*np.pi, goal_state[1], facecolor='green', s=5)

        # ax.set_aspect('equal')
        plt.xlabel('$\\theta$')
        plt.ylabel('$\dot{\\theta}$')
        duration += (end_time-start_time)
        plt.title('RG RRT Tree after %.2f seconds (explored %d nodes)' %(duration, rrt.node_tally))
        plt.savefig('RG_RRT_Pendulum_'+experiment_name+'/%.2f_seconds_tree.png' % duration, dpi=500)
        # plt.show()
        plt.xlim([-4, 4])
        plt.ylim([-10,10])
        plt.clf()
        plt.close()

        # # Plot explored reachable sets
        # FIXME: Handle degenerated reachable set
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # fig, ax = visualize_2D_AH_polytope(reachable_polytopes, fig=fig, ax=ax)
        #
        # ax.scatter(initial_state[0], initial_state[1], facecolor='red', s=5)
        # ax.scatter(goal_state[0], goal_state[1], facecolor='green', s=5)
        # ax.scatter(goal_state[0]-2*np.pi, goal_state[1], facecolor='green', s=5)
        #
        # # ax.set_aspect('equal')
        # plt.xlabel('$x$')
        # plt.ylabel('$\dot{x}$')
        # duration += (end_time-start_time)
        # plt.title('RRT Tree after %.2f seconds (explored %d nodes)' %(duration, len(polytope_reachable_sets)))
        # plt.savefig('RRT_Pendulum_'+experiment_name+'/%.2f_seconds_reachable_sets.png' % duration, dpi=500)
        # # plt.show()
        # plt.clf()
        # plt.close()

        if found_goal:
            break

if __name__=='__main__':
    for i in range(1):
        test_pendulum_planning()