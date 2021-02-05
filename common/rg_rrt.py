import numpy as np
from rtree import index
from r3t.common.basic_rrt import *
from timeit import default_timer


class RGRRT(BasicRRT):
    def __init__(self, root_state, sampler, reached_goal_function, plan_collision_free_path_towards, state_tree=StateTree(), rewire_radius = None):
        self.root_node = Node(root_state, cost_from_parent=0, true_dynamics_path=[root_state, root_state])
        self.root_id = hash(str(root_state))
        self.state_dim = root_state[0]
        self.sampler = sampler
        self.reached_goal = reached_goal_function
        self.goal_state = None
        self.goal_node = None
        self.state_tree = state_tree
        self.state_tree.insert(self.root_id,self.root_node.state)
        self.state_to_node_map = dict()
        self.true_path_map = dict()
        self.state_to_node_map[self.root_id] = self.root_node
        self.node_tally = 0
        self.rewire_radius=rewire_radius
        self.plan_collision_free_path_towards=plan_collision_free_path_towards
        self.reachable_set_tree=StateTree()
        self.reachabe_state_to_node_map=dict()
        reachable_states, true_dynamics_path = self.get_reachable_states(root_state)
        for i, rs in enumerate(reachable_states):
            rs_hash = hash(str(rs))
            self.reachable_set_tree.insert(rs_hash, rs)
            self.reachabe_state_to_node_map[rs_hash]=self.root_node
            self.true_path_map[str(root_state)+str(rs)]=true_dynamics_path[i]

    def create_child_node(self, parent_node, child_state, cost_from_parent, path_from_parent, true_dynamics_path):
        '''
        Given a child state reachable from a parent node, create a node with that child state
        :param parent_node: parent
        :param child_state: state inside parent node's reachable set
        :param cost_from_parent: FIXME: currently unused
        :param path_from_parent: FIXME: currently unused
        :return:
        '''
        # Update the nodes
        # compute the cost to go and path to reach from parent
        # if cost_from_parent is None or path_from_parent is None:
        # assert (parent_node.reachable_set.contains(child_state))
        # construct a new node
        new_node = Node(child_state, parent=parent_node, path_from_parent=path_from_parent, cost_from_parent=cost_from_parent, \
                        true_dynamics_path=true_dynamics_path)
        parent_node.children.add(new_node)
        reachable_states, true_dynamics_path = self.get_reachable_states(child_state)
        for i, rs in enumerate(reachable_states):
            rs_hash = hash(str(rs))
            self.reachable_set_tree.insert(rs_hash, rs)
            self.reachabe_state_to_node_map[rs_hash]=new_node
            self.true_path_map[str(child_state)+str(rs)]=true_dynamics_path[i]
        return new_node

    def get_reachable_states(self, state):
        raise NotImplementedError

    def extend(self, new_state, nearest_node, explore_deterministic_next_state=False):
        # test for possibility to extend
        cost_to_go, end_state, best_states_list = self.plan_collision_free_path_towards(nearest_node.state, new_state)
        if end_state is None:
            return False, None
        new_node = self.create_child_node(nearest_node, end_state, cost_to_go, end_state, true_dynamics_path=best_states_list)
        return True, new_node

    def force_extend(self, new_state, nearest_node, true_dynamics_path):
        # test for possibility to extend
        new_node = self.create_child_node(nearest_node, new_state, np.linalg.norm(new_state-nearest_node.state), new_state,true_dynamics_path)
        return True, new_node

    def build_tree_to_goal_state(self, goal_state, allocated_time=20, stop_on_first_reach=False, rewire=False,
                                     explore_deterministic_next_state=True, max_nodes_to_add=int(1e9)):
        start = default_timer()
        self.goal_state = goal_state
        # For cases where root node can lead directly to goal
        if self.reached_goal(self.root_node.state, goal_state):
            goal_node = self.create_child_node(self.root_node, goal_state)
            # if rewire:
            #     self.rewire(goal_node)
            self.goal_node=goal_node

        while True:
            if stop_on_first_reach:
                if self.goal_node is not None:
                    print('Found path to goal with cost %f in %f seconds after exploring %d nodes' % (self.goal_node.cost_from_root,
                    default_timer() - start, self.node_tally))
                    return self.goal_node
            if default_timer()-start>allocated_time:
                if self.goal_node is None:
                    print('Unable to find path within %f seconds!' % (default_timer() - start))
                    return None
                else:
                    print('Found path to goal with cost %f in %f seconds after exploring %d nodes' % (self.goal_node.cost_from_root,
                    default_timer() - start, self.node_tally))
                    return self.goal_node

            #sample the state space
            state_sample = self.sampler()
            nearest_reachable_state = self.reachable_set_tree.find_nearest(state_sample)
            nearest_node = self.reachabe_state_to_node_map[hash(str(nearest_reachable_state))]
            new_state_id = hash(str(nearest_reachable_state))
            if new_state_id in self.state_to_node_map:
                continue
            true_dynamics_path = self.true_path_map[str(nearest_node.state)+str(nearest_reachable_state)]
            is_extended, new_node = self.force_extend(nearest_reachable_state, nearest_node, true_dynamics_path)
            if not is_extended: #extension failed
                print('Warning: extension failed')
                continue
            # try:
            #     assert(new_state_id not in self.state_to_node_map)
            # except:
            #     print('State id hash collision!')
            #     print('Original state is ', self.state_to_node_map[new_state_id].state)
            #     print('Attempting to insert', new_node.state)
            #     raise AssertionError
            self.state_tree.insert(new_state_id, new_node.state)
            self.state_to_node_map[new_state_id] = new_node

            # print('snm', len(self.state_to_node_map))
            # print(len(self.state_tree.state_id_to_state))
            self.node_tally = len(self.state_to_node_map)
            # TODO
            #rewire the tree
            # if rewire:
            #     self.rewire(new_node)
            #In "find path" mode, if the goal is in the reachable set, we are done
            if self.reached_goal(new_node.state, goal_state): #FIXME: support for goal region
                # add the goal node to the tree
                is_extended, goal_node = self.extend(goal_state, new_node)
                # TODO
                # if rewire:
                #     self.rewire(goal_node)
                if is_extended:
                    self.goal_node=goal_node
