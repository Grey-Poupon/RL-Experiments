import numpy as np
import math
import random

class Item:
    def __init__(self, resource, TD_error= 200):
        self.resource = resource
        self.TD_error = TD_error
        self.priority = None
        self.idx = None
        self.tree = None
        self.recalc_priority = False

    def __eq__(self, other):
        return type(self) == type(other) and np.array_equal(self.resource , other.resource) and self.TD_error == other.TD_error and\
               self.get_priority() == other.get_priority() and self.idx == other.idx

    def get_priority(self):
        if self.recalc_priority:
            self.calc_rank()
            self.recalc_priority = False
            self.tree.update_sum_value(self.tree.get_parent(self.idx))

        return self.priority

    def update_TD_error(self, TD_error):
        self.TD_error = TD_error

    def set_idx(self, idx):
        self.idx = idx
        self.recalc_priority = True


    def calc_rank(self):
        rank = float(self.tree.get_right_most_node(self.tree.layers) + 1 - self.idx)
        self.priority = 1 / rank

    def get_sample_prob(self):
        return self.get_priority() / self.tree.get_sum_priority()


class SumTree:
    def __init__(self, size, items):

        # Get the size needed for the binary tree
        self.layers = np.log2(size)
        self.layers = math.ceil(self.layers) + 1

        self.layers = int(self.layers)
        self.size = int(math.pow(2, self.layers)) - 1

        # Init tree
        self.tree = list(np.zeros(self.size))

        # Insert items into leaves / Build tree
        items.sort(key=lambda x: abs(x.TD_error), reverse=False)
        for i in items:
            i.tree = self
        self.insert_leaves(items)

        self.first_empty_leaf = self.get_left_most_node(self.layers) + len(items)

        self.empty_leaves = math.pow(2, self.layers) - len(items)

    def __eq__(self, other):

        return type(self) == type(other) and (self.tree == other.tree) and  self.layers == other.layers and \
               self.empty_leaves == other.empty_leaves and self.first_empty_leaf == other.first_empty_leaf and  self.size == other.size

    def get_max_weight(self, beta):
        # w= (N·P(j))−β
        #  P(j) = Prior/SumPrior
        #  Prior = 1/rank
        #  Max Rank = N
        #  Therefore Max P(j) = 1/N·SumPrior
        #  N·MaxP(j)  = 1/SumPrior
        # (N·P(j))−β  = 1/(N·P(j))β
        #             = 1/1/(SumPrior)β
        #             = (SumPrior)β
        # MaxWeight = (SumPrior)β
        return math.pow(self.get_sum_priority(), beta)

    # Over time our heap stops looking like a sorted array and we have to resort it
    def sort_tree(self):
        leaves = self.tree[self.get_left_most_node(self.layers): self.get_right_most_node(self.layers) + 1]
        leaves.sort(key=lambda x: abs(x.TD_error), reverse=False)
        self.insert_leaves(leaves)

    # Proportional prioritization as per PER
    def get_minibatch(self, batch_size):
        item_idx = self.get_left_most_node(self.layers)
        end_item = self.get_right_most_node(self.layers)

        items = []
        step = self.tree[end_item].get_priority() / batch_size
        min_ = 0

        for i in range(batch_size):
            sub_step = random.random() * step
            sample = min_ + sub_step

            ## get item
            while item_idx < end_item and self.tree[item_idx].get_priority() < sample:
                item_idx += 1

            items.append(self.tree[item_idx])
            min_ += step
        return items

    def get_sum_priority(self):
        if self.tree[0] == 0:
            raise EnvironmentError
        return self.tree[0]

    ##
    # Add item to tree
    ##
    def add_item(self, item):
        if item.tree is None:
            item.tree = self

        free_idx = self.first_empty_leaf
        left_idx = self.get_left_most_node(self.layers)
        # Find empty leaf
        while free_idx < self.size and self.tree[free_idx] != 0:
            free_idx += 1

        # If tree is full delete "smallest" leaf
        if free_idx >= self.size:

            # shift leaves 1 to the left

            self.tree[left_idx:-1] = self.tree[left_idx + 1:]
            # Set item as newest/largest leaf
            self.tree[-1] = item
            # update parent -- Only necessary if p = |TD_error| as rank won't change respective to parents
            # self.update_sum_value(self.get_parent(item.idx))

            # update idx for moved items
            for idx, leaf in enumerate(self.get_leaves()):
                leaf.set_idx(idx + left_idx)
        else:
            # Set leaf & do updates
            self.tree[free_idx] = item

            for idx, leaf in enumerate(self.get_leaves()):
                leaf.set_idx(idx + left_idx)

            self.update_sum_value(self.get_parent(free_idx))
            # Update leaf count
            self.empty_leaves -= 1
            self.first_empty_leaf = free_idx + 1

    ##
    # Tree Building
    ##
    def insert_leaves(self, leaves):
        if len(leaves) == 0:
            return

        # Get left-most leaf
        left_idx = self.get_left_most_node(self.layers)

        # Insert leaves
        for idx in range(len(leaves)):
            if left_idx + idx > self.size - 1:
                break
            self.tree[left_idx + idx] = leaves[idx]
            self.tree[left_idx + idx].set_idx(left_idx + idx)



        # Update sums
        self.init_tree_sums()

    ##
    # Value Updates
    ##

    def update_leaf_priority(self, idx, priority):
        self.tree[idx].priority = priority
        self.update_sum_value(self.get_parent(idx))

    def update_sum_value(self, idx):

        if self.tree[idx] == 0:
            self.tree[idx] = Item(0, 0)
            self.tree[idx].idx = idx

        l_idx = self.get_left_child(idx)
        r_idx = self.get_right_child(idx)

        # If we are summing leaves we need the .priority else the item is the priority
        if self.get_left_most_node(self.layers) <= l_idx:

            left = 0 if self.tree[l_idx] == 0 else self.tree[l_idx].get_priority()
            right = 0 if self.tree[r_idx] == 0 else self.tree[r_idx].get_priority()

        else:
            left = 0 if self.tree[l_idx] == 0 else self.tree[l_idx]
            right = 0 if self.tree[r_idx] == 0 else self.tree[r_idx]

        self.tree[idx] = left + right

        if idx != 0:
            self.update_sum_value(self.get_parent(idx))

    def update_layer_of_values(self, layer):
        node_count = math.pow(2, layer)
        idx = self.get_left_most_node(layer)

        for i in range(idx, idx + node_count):
            self.update_sum_value(idx)

    def init_tree_sums(self):

        for layer in range(self.layers - 1, 0, -1):

            left_node  = self.get_parent(self.get_left_most_node(layer+1))
            right_node = self.get_parent(self.get_right_most_node(layer+1))

            if layer == self.layers - 1:
                for i in range(left_node, right_node + 1):
                    left_idx    = self.get_left_child(i)
                    right_idx   = self.get_right_child(i)
                    left_child  = self.tree[left_idx]
                    right_child = self.tree[right_idx]
                    left_priority = left_child.get_priority()   if left_child  != 0 else 0
                    right_priority = right_child.get_priority() if right_child != 0 else 0
                    self.tree[i] = left_priority + right_priority

            else:
                for i in range(left_node, right_node + 1):

                    self.tree[i] = self.tree[self.get_left_child(i)] + self.tree[self.get_right_child(i)]


    ##
    # Indexing
    ##

    def get_left_child(self, idx):
        return int((2 * idx) + 1)

    def get_right_child(self, idx):
        return int((2 * idx) + 2)

    def get_parent(self, idx):
        return int((idx - 1) // 2)

    def get_left_most_node(self, layer):
        return int(math.pow(2, layer-1)-1)

    def get_right_most_node(self, layer):
        # get right most node
        idx = self.size - 1
        for i in range(self.layers - layer):
            idx = self.get_parent(idx)

        # find non-zero value
        while idx > 1 and self.tree[idx] == 0:
            idx -= 1
        return idx

    def get_num_leaves(self):

        layer = self.layers
        return self.get_right_most_node(layer) - self.get_left_most_node(layer) + 1

    def get_leaves(self):
        l = self.get_left_most_node(self.layers)
        r = self.get_right_most_node(self.layers)
        return self.tree[l:r+1]

    def save(self):

        leaves = self.get_leaves()
        states     = np.empty((len(leaves), 4,84,84,1), dtype=np.int)
        actions    = np.empty(len(leaves), dtype=np.int)
        rewards    = np.empty(len(leaves), dtype=np.float64)
        new_states = np.empty((len(leaves), 4,84,84,1), dtype=np.int)
        terminals  = np.empty(len(leaves), dtype=np.bool)
        TD_errors  = np.empty(len(leaves))

        for idx, leaf in enumerate(leaves):
            states[idx]     = leaf.resource[0]
            actions[idx]    = leaf.resource[1]
            rewards[idx]    = leaf.resource[2]
            new_states[idx] = leaf.resource[3]
            terminals[idx]  = leaf.resource[4]
            TD_errors[idx]  = leaf.TD_error

        return states, actions, rewards, new_states, terminals, TD_errors


class PEReplayMemory(object):
    """Replay Memory that stores the last size=1,000,000 transitions"""

    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size:                 Number of stored transitions
            frame_height:         Height of a frame
            frame_width:          Width  of a frame
            agent_history_length: Number of frames stacked together to create a state
            batch_size:           Number of transitions  in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.tree = SumTree(self.size, [])
        self.sort_timer = self.size

    # Load from a saved file
    def load(self, file):
        state     = file['arr_0']
        action    = file['arr_1']
        reward    = file['arr_2']
        new_state = file['arr_3']
        terminal  = file['arr_4']
        TD_error  = file['arr_5']

        items = []
        for i in range(len(TD_error)):
            items.append(Item([state[i], action[i], reward[i], new_state[i], terminal[i]], TD_error[i]))

        self.tree = SumTree(self.size, items)

    def add_experience(self, action, state, new_state, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            state: A (84, 84, 4) state that the agent interpreted from (Frames of Game)
            reward: A float determining the reward the agent received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if state.shape != (self.frame_height, self.frame_width, self.agent_history_length):
            raise ValueError('Dimension of frame is wrong!')

        resource = [state, action, reward, new_state, terminal]
        memory = Item(resource, None)
        memory.tree = self.tree

        self.tree.add_item(memory)

        # Sort Tree if needed
        self.sort_timer -= 1
        if self.sort_timer == 0:
            self.sort()
            self.sort_timer = self.size

    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.tree.get_num_leaves() < self.batch_size:
            raise ValueError('Not enough memories to get a minibatch')

        transitions = self.tree.get_minibatch(self.batch_size)

        states, actions, rewards, new_states, terminal_flags, probabilties = [], [], [], [], [], []

        for t in transitions:
            states.append(t.resource[0])
            actions.append(t.resource[1])
            rewards.append(t.resource[2])
            new_states.append(t.resource[3])
            terminal_flags.append(t.resource[4])
            probabilties.append(t.get_sample_prob())

        return np.array(states), np.array(actions), np.array(rewards), np.array(new_states), np.array(terminal_flags), np.array(probabilties)

    def save(self):
        return self.tree.save()

    def sort(self):
        self.tree.sort_tree()
