import numpy as np
import math
import random
import time
from collections import deque
from bisect import bisect_left

class Item:
    def __init__(self, resource, TD_error= 200):
        self.resource = resource
        self.TD_error = TD_error
        self.priority = None
        self.idx = None
        self.tree = None
        self.rank = None

    def __eq__(self, other):
        return type(self) == type(other) and np.array_equal(self.resource , other.resource) and self.TD_error == other.TD_error and\
               self.get_priority() == other.get_priority() and self.idx == other.idx

    def get_priority(self):
        self.calc_rank()
        return self.priority

    def get_sample_prob(self):
        return self.get_priority()/self.tree.get_sum_priority()

    def update_TD_error(self, TD_error):
        self.TD_error = TD_error

    def calc_rank(self):
        self.rank = self.get_idx() + 1
        self.priority = 1 / self.rank

    def get_idx(self):
        return self.idx - self.tree.idx_shift

    def __lt__(self, other):
        return self.get_priority() < other

class SumTree:
    def __init__(self, size, items):

        # Get the size needed for the binary tree

        self.size = size
        self.empty_leaves = self.size - len(items)
        self.idx_shift = 0

        # Insert items into leaves / Build tree
        items.sort(key=lambda x: abs(x.TD_error), reverse=True)

        # Init tree
        self.tree = deque(items, maxlen=self.size)

        for idx, i in enumerate(self.tree):
            i.tree = self
            i.idx = idx



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
        if beta == 1:
            return self.get_sum_priority()
        return math.pow(self.get_sum_priority(), beta)

    # Over time our heap stops looking like a sorted array and we have to resort it
    def sort_tree(self):
        self.tree.sort(key=lambda x: abs(x.TD_error), reverse=False)

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

    """
    To sample a minibatch of size k, the range[0,P_total] is divided equally into k ranges.
    Next, a value is uniformly sampled from each range.
    Finally the transitions that correspond to each of these sampled values are retrieved from the tree
    """
    def get_minibatch2(self, batch_size):
        leaves = self.get_leaves()

        lp = self.get_sum_priority()
        step = lp/batch_size

        batch = []
        val = 0
        l_idx = 0

        for i in range(1, batch_size+1):
            val += step
            r_idx = self.take_closest(leaves, val)

            random_idx = random.randrange(l_idx, r_idx) if l_idx!=r_idx else l_idx
            batch.append(leaves[random_idx])

            l_idx = r_idx

        return batch

    def take_closest(self, leaves, val):
        pos = bisect_left(leaves, val)
        if pos == 0:
            return 0
        return pos - 1

    def H(self, n):
        """Returns an approximate value of n-th harmonic number.

           http://en.wikipedia.org/wiki/Harmonic_number
        """
        # Euler-Mascheroni constant
        gamma = 0.57721566490153286060651209008240243104215933593992
        return gamma + math.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)

    def get_sum_priority(self):
        return self.H(self.get_num_leaves())
        # if self.items_added or self.values_changes:
        #     self.full_tree_sum_update()
        #     self.items_added = False
        #     self.values_changes = False
        #
        # if self.tree[0] == 0:
        #     raise EnvironmentError
        # return self.tree[0]

    ##
    # Add item to tree
    ##
    def add_item(self, item):
        if item.tree is None:
            item.tree = self
        if self.empty_leaves == 0:
            self.tree.append(item)
            self.idx_shift += 1
            item.idx = self.size + self.idx_shift
        else:
            self.tree.append(item)
            item.idx = self.size - self.empty_leaves + self.idx_shift
            self.empty_leaves -= 1

    ##
    # Indexing
    ##

    def get_left_most_node(self):
        return 0

    def get_right_most_node(self):
        # Get right most leaf
        idx = self.get_left_most_node(self.layers) + self.get_num_leaves() - 1

        return idx

    def get_num_leaves(self):
        return int(self.size - self.empty_leaves)

    def get_leaves(self):
        return self.tree

    def save(self):

        leaves = self.get_leaves()
        states     = np.empty((len(leaves), 84,84,4), dtype=np.int)
        actions    = np.empty(len(leaves), dtype=np.int)
        rewards    = np.empty(len(leaves), dtype=np.float64)
        new_states = np.empty((len(leaves), 84,84,1), dtype=np.int)
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

        transitions = self.tree.get_minibatch2(self.batch_size)

        states, actions, rewards, new_states, terminal_flags, probabilties = [], [], [], [], [], []

        for t in transitions:
            states.append(t.resource[0])
            actions.append(t.resource[1])
            rewards.append(t.resource[2])
            new_states.append(np.concatenate([t.resource[0][..., 1:], t.resource[3]], axis=-1))
            terminal_flags.append(t.resource[4])
            probabilties.append(t.get_sample_prob())

        return np.array(states), np.array(actions), np.array(rewards), np.array(new_states), np.array(terminal_flags), np.array(probabilties)

    def save(self):
        return self.tree.save()

    def sort(self):
        self.tree.sort_tree()
