import pydrake
from r3t.common.basic_rrt import *
from r3t.common.rg_rrt import *
#todo change al_control source
from polytope_symbolic_system.examples.pendulum import Pendulum
from r3t.symbolic_system.examples.al_control_step.alcontrol import alcontrol_step

class SymbolicSystem_RGRRT(RGRRT):
    def __init__(self, sys, sampler, step_size, reached_goal_function, linearize=True):
        self.sys = sys
        self.step_size = step_size
        self.reached_goal_function=reached_goal_function
        self.nonlinear_dynamics_step_size = Pendulum.time_step
        self.linearize = linearize
        # self.posi_list = []
        # self.orie_list = []
        # self.new_postion = np.zeros([3,1])
        def plan_collision_free_path_towards(nearest_state, new_state):
            # possible_inputs = np.linspace(*sys.input_limits[:, 0], num=3)
            best_input = None
            best_new_state = None
            best_distance = np.inf
            best_states_list = None
            # for input in possible_inputs: #todo lqr
            states_list = [np.atleast_1d(nearest_state)]
            state = np.atleast_1d(nearest_state)
            # for step in range(int(self.step_size / self.nonlinear_dynamics_step_size)):
            # #     # todo change alcontrol_step
            #     state = self.sys.lqr_forward_step(starting_state=state, linearize=True,
            #                                   modify_system=False, return_as_env=False,
            #                                   step_size=self.nonlinear_dynamics_step_size) #u=np.atleast_1d(input),
            #     states_list.append(state)
                #fixme cost=||distance||?
            state, al_states_list = alcontrol_step(starting_state=state, target_state=new_state,
                                                                      states_list=states_list, sz=self.step_size, ndsz=self.nonlinear_dynamics_step_size)
            new_distance = np.linalg.norm(new_state - state)
            # print(new_distance)
            if new_distance<best_distance:
                best_input = input
                best_new_state=state
                best_distance = new_distance
                best_states_list = al_states_list
            return best_distance, best_new_state, best_states_list

        RGRRT.__init__(self,self.sys.get_current_state(),sampler,reached_goal_function, plan_collision_free_path_towards)

    def get_reachable_states(self, state):
        possible_inputs = np.linspace(*self.sys.input_limits[:, 0], num=3) #[-1.  0.  1.]
        possible_new_states = []
        true_dynamics_paths = []
        for input in possible_inputs:
            state_list = [state]
            s = state
            for i in range(int(self.step_size/self.nonlinear_dynamics_step_size)):
                # s, al_states_list = quad_alcontrol_step(starting_state=s, target_state=new_state, states_list=states_list)
                 s = self.sys.forward_step(starting_state=np.atleast_1d(s),
                                                        u=np.atleast_1d(input), modify_system=False,
                                                        return_as_env=False, step_size=self.nonlinear_dynamics_step_size)
                 state_list.append(s)
            possible_new_states.append(s)
            true_dynamics_paths.append(state_list)
        return possible_new_states,true_dynamics_paths