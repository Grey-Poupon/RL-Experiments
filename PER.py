TRAIN = True

ENV_NAME = 'BreakoutDeterministic-v4'
#ENV_NAME = 'PongDeterministic-v4'
import math
import os
import random
import gym
import tensorflow as tf
import numpy as np
from datetime import datetime


class FrameProcessor(object):
    """Resizes and converts RGB Atari frames to grayscale"""
    def _process(self, frame, frame_height, frame_width):
            processed = tf.image.rgb_to_grayscale(frame)
            processed = tf.image.crop_to_bounding_box(processed, 34, 0, 160, 160)
            return tf.image.resize(processed, [frame_height, frame_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    def __init__(self, frame_height=84, frame_width=84):
        """
        Args:
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.process = tf.function(self._process)


    def __call__(self, frame):
        """
        Args:
            frame: A (210, 160, 3) frame of an Atari game in RGB
        Returns:
            A processed (84, 84, 1) frame in grayscale
        """
        return self.process(frame, self.frame_height, self.frame_width)


class DQN(object):
    """Implements a Deep Q Network"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, n_actions, hidden=1024, learning_rate=0.00001,
                 frame_height=84, frame_width=84, agent_history_length=4):
        """
        Args:
            n_actions: Integer, number of possible actions
            hidden: Integer, Number of filters in the final convolutional layer.
                    This is different from the DeepMind implementation
            learning_rate: Float, Learning rate for the Adam optimizer
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
        """
        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length

        self.input = tf.keras.Input(shape=[self.frame_height, self.frame_width, self.agent_history_length], dtype=tf.dtypes.float32)
        # Normalizing the input
        self.inputscaled = tf.math.divide(self.input, tf.dtypes.cast(255, tf.dtypes.float32))

        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=[8, 8], strides=[4, 4],
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1', dtype=tf.dtypes.float32)(self.inputscaled)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=[4, 4], strides=[2, 2],
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2', dtype=tf.dtypes.float32)(self.conv1)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=[3, 3], strides=[1, 1],
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3', dtype=tf.dtypes.float32)(self.conv2)
        self.conv4 = tf.keras.layers.Conv2D(
            filters=hidden, kernel_size=[7, 7], strides=[1, 1],
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4', dtype=tf.dtypes.float32)(self.conv3)

        # Splitting into value and advantage stream
        self.valuestream, self.advantagestream = tf.keras.layers.Lambda(lambda tensor: tf.split(tensor, 2, 3))(self.conv4)

        self.valuestream = tf.keras.layers.Flatten()(self.valuestream)
        self.advantagestream = tf.keras.layers.Flatten()(self.advantagestream)

        self.advantage = tf.keras.layers.Dense(units=self.n_actions,
                                           kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                           name='advantage')(self.advantagestream)

        self.value = tf.keras.layers.Dense(units=1,
                                           kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                           name='value')(self.valuestream)

        # Combining value and advantage into Q-values
        def recombine(x):
            value = x[0]
            advantage = x[1]
            return tf.add(value , tf.subtract(advantage, tf.reduce_mean(input_tensor=advantage, axis=1, keepdims=True)))

        self.q_values = tf.keras.layers.Lambda(recombine)([self.value, self.advantage])
        self.model = tf.keras.Model(inputs=self.input, outputs=self.q_values)
        self.model.compile( optimizer='Adam', loss=tf.keras.losses.Huber())


class ExplorationExploitationScheduler(object):
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""

    def __init__(self, DQN, n_actions, eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
                 eps_evaluation=0.0, eps_annealing_frames=1000000,
                 replay_memory_start_size=50000, max_frames=25000000):
        """
        Args:
            DQN: A DQN object
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_frames = max_frames

        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
                    self.max_frames - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames

        self.DQN = DQN

    def get_epsilon(self, frame_number, evaluation=False):
        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2 * frame_number + self.intercept_2
        return eps

    def get_action(self, frame_number, state, evaluation=False):
        """
        Args:
            frame_number: Integer, number of the current frame
            state: A (84, 84, 4) sequence of frames of an Atari game in grayscale
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions - 1 determining the action the agent perfoms next
        """

        if np.random.rand(1) < self.get_epsilon(frame_number, evaluation):
            return np.random.randint(0, self.n_actions)

        state = tf.cast(tf.expand_dims(state, 0), tf.dtypes.float32)
        values = self.DQN.model(state)
        argmax = np.argmax(values, axis=-1)
        return argmax


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

    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.tree.get_num_leaves() < self.batch_size:
            raise ValueError('Not enough memories to get a minibatch')

        return self.tree.get_minibatch(self.batch_size)

    def save(self):
        return self.tree.save()

    def sort(self):
        self.tree.sort_tree()


def learn(PER_memory, main_dqn, target_dqn, batch_size, gamma):
    """
    Args:
        replay_memory: A ReplayMemory object
        main_dqn: A DQN object
        target_dqn: A DQN object
        batch_size: Integer, Batch size
        gamma: Float, discount factor for the Bellman equation
    Returns:
        loss: The loss of the minibatch, for tensorboard
    Draws a minibatch from the replay memory, calculates the
    target Q-value that the prediction Q-value is regressed to.
    Then a parameter update is performed on the main DQN.
    """
    # Draw a minibatch from the replay memory
    transitions = PER_memory.get_minibatch()
    states, actions, rewards, new_states, terminal_flags, probabilties = [],[],[],[],[],[]

    for t in transitions:
        states.append(t.resource[0])
        actions.append(t.resource[1])
        rewards.append(t.resource[2])
        new_states.append(t.resource[3])
        terminal_flags.append(t.resource[4])
        probabilties.append(t.get_sample_prob())

    matches =0
    all_matches = True
    for i in range(1,len(states)):
        matches += np.equal(states[i], states[i-1]).all()
        all_matches = all_matches and np.equal(states[i], states[i-1]).all()

    print(all_matches, matches)
    states, actions, rewards, new_states, terminal_flags, probabilties = np.array(states), np.array(actions), np.array(rewards), np.array(new_states), np.array(terminal_flags), np.array(probabilties)

    # The main network estimates which action is best (in the next
    # state s', new_states is passed!)
    # for every transition in the minibatch

    values = main_dqn.model(tf.cast(new_states, tf.dtypes.float32))
    arg_q_max = tf.squeeze(tf.image.rot90(tf.expand_dims([tf.range(len(values)-1,-1,-1, dtype=tf.int64), tf.argmax(values, axis=-1)] ,-1)))
    # The target network estimates the Q-values (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    q_vals = target_dqn.model(tf.cast(new_states, tf.dtypes.float32))

    double_q = tf.gather_nd(q_vals, arg_q_max)
    # We use an importance sampling weight to reduce the bias introduces using PER
    # weight = (N·P(j))−β / Max Weight
    N = PER_memory.tree.get_num_leaves()
    P = probabilties
    #M = 1
    B = PER_memory.tree.get_max_weight(1)

    # Calculcate weight as per PER
    w = tf.math.pow((N * P), -B)
    # Normalise using max Weight however max=1 as B=1 therofre is note needed
    #importace_sampling_weight = tf.cast(tf.divide(w , M), dtype=tf.float32)
    importace_sampling_weight = tf.cast(w, dtype=tf.float32)


    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = (rewards + (gamma * double_q * (1 - terminal_flags))) * importace_sampling_weight
    # Gradient descend step to update the parameters of the main network
    x = tf.one_hot(actions, main_dqn.n_actions, dtype=tf.float32)
    # Have to stack here so that x.shape == y.shape for multiplcation
    y = tf.stack([target_q, target_q, target_q, target_q], axis=1)
    wanted_Qs = tf.multiply(y, x)
    # Gradient descend step to update the parameters of the main network
    loss =  main_dqn.model.fit(states, wanted_Qs, verbose=0).history['loss']

    # Calc TD_Error, update items
    expected_q = target_dqn.model(tf.cast(states, tf.dtypes.float32))
    expected_q = tf.gather_nd(expected_q, arg_q_max)
    TD_Error = target_q - expected_q

    for i in range(len(TD_Error)):
        transitions[i].update_TD_error(TD_Error[i])

	return loss, np.array(TD_Error)


class TargetNetworkUpdater(object):
    """Copies the parameters of the main DQN to the target DQN"""

    def __init__(self, main_dqn_vars, target_dqn_vars):
        """
        Args:
            main_dqn_vars: A list of tensorflow variables belonging to the main DQN network
            target_dqn_vars: A list of tensorflow variables belonging to the target DQN network
        """
        self.main_dqn_vars = main_dqn_vars
        self.target_dqn_vars = target_dqn_vars

    def _update_target_vars(self):
        for i, var in enumerate(self.main_dqn_vars):
            self.target_dqn_vars[i].assign(var.value())

    def __call__(self):
        """
        Assigns the values of the parameters of the main network to the
        parameters of the target network
        """
        tf.function(self._update_target_vars())


class Atari(object):
    """Wrapper for the environment provided by gym"""

    def __init__(self, envName, no_op_steps=10, agent_history_length=4):
        self.env = gym.make(envName)
        self.process_frame = FrameProcessor()
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length

    def reset(self, evaluation=False):
        """
        Args:
            evaluation: A boolean saying whether the agent is evaluating or training
        Resets the environment and stacks four frames ontop of each other to
        create the first state
        """
        frame = self.env.reset()
        self.last_lives = 0
        terminal_life_lost = True  # Set to true so that the agent starts
        # with a 'FIRE' action when evaluating
        if evaluation:
            for _ in range(random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1)  # Action 'Fire'
        processed_frame = self.process_frame(frame)  # (★★★)
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)

        return terminal_life_lost

    def step(self, action):
        """
        Args:
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)  # (5★)

        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_new_frame = self.process_frame(new_frame)  # (6★)
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2)  # (6★)
        self.state = new_state

        return new_state, reward, terminal, terminal_life_lost, new_frame


def clip_reward(reward):
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1



# Control parameters
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
EVAL_FREQUENCY = 50000          # Number of frames the agent sees between evaluations
EVAL_STEPS = 10000               # Number of frames for one evaluation
NETW_UPDATE_FREQ = 10000         # Number of chosen actions between updating the target network.
                                 # According to Mnih et al. 2015 this is measured in the number of
                                 # parameter updates (every four actions), however, in the
                                 # DeepMind code, it is clearly measured in the number
                                 # of actions the agent choses
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions,
                                 # before the agent starts learning
MAX_FRAMES = 30000000            # Total number of frames the agent sees
MEMORY_SIZE = 300000             # Number of transitions stored in the replay memory
NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                 # evaluation episode
UPDATE_FREQ = 4                  # Every four actions a gradient descend step is performed
HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output
                                 # has the shape (1,1,1024) which is split into two streams. Both
                                 # the advantage stream and value stream have the shape
                                 # (1,1,512). This is slightly different from the original
                                 # implementation but tests I did with the environment Pong
                                 # have shown that this way the score increases more quickly
LEARNING_RATE = 0.00001          # Set to 0.00025 in Pong for quicker results.
                                 # Hessel et al. 2017 used 0.0000625
BS = 32                          # Batch size

PATH = "/content/gdrive/My Drive/Models/PER_Breakout/"                 # checkpoints will be saved here
RUNID = 'run_1'

atari = Atari(ENV_NAME, NO_OP_STEPS)

print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                        atari.env.unwrapped.get_action_meanings()))


# main DQN and target DQN networks:
MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE)
TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)

# Trainalbe variables so we can crossover later
MAIN_DQN_VARS = MAIN_DQN.model.trainable_variables
TARGET_DQN_VARS = TARGET_DQN.model.trainable_variables


def train():
    """Contains the training and evaluation loops"""
    replay_memory = PEReplayMemory(size=MEMORY_SIZE, batch_size=BS)  # (★)

    explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n,
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
        max_frames=MAX_FRAMES)

    # Load Memory  & Network Weights
    # replay_memory.load(np.load(PATH+"Memory.npy",allow_pickle=True))
    print("Imported Memory")
    # MAIN_DQN.model.load_weights(PATH + "")
    update_networks = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)

    frame_number = 0
    rewards = []
    log_list = []
    run = 0

    while frame_number < MAX_FRAMES:  #

        epoch_frame = 0
        while epoch_frame < EVAL_FREQUENCY:
            run += 1
            terminal_life_lost = atari.reset()
            episode_reward_sum = 0
            for _ in range(MAX_EPISODE_LENGTH):

                state =atari.state
                action = explore_exploit_sched.get_action(frame_number, state)

                processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(action)
                frame_number += 1
                epoch_frame += 1
                episode_reward_sum += reward

                # Clip the reward
                clipped_reward = clip_reward(reward)

                # Store transition in the replay memory
                replay_memory.add_experience(   action=action,
                                                new_state=processed_new_frame,
                                                state=state,
                                                reward=clipped_reward,
                                                terminal=terminal_life_lost)

                if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                    loss, TD_error = learn(replay_memory, MAIN_DQN, TARGET_DQN,
                                 BS, gamma=DISCOUNT_FACTOR)
                    log_list.append([loss,TD_error])
                if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                    update_networks()  # (9★)

                if terminal:
                    #print("Run: " + str(run) + "  Reward: " + str(episode_reward_sum) + "  Explore Rate: " + str( explore_exploit_sched.get_epsilon(frame_number)) + "  Frame Count: " + str(frame_number))
                    terminal = False
                    break

            rewards.append(episode_reward_sum)

        # Save the network parameters + Memory
        MAIN_DQN.model.save_weights(PATH + str(frame_number))
        replay_memory.sort()
        save_states, save_actions, save_rewards, save_new_states, save_terminals, save_TD_errors = replay_memory.save()
        np.save(PATH + "Logs "+ str(frame_number), log_list)
        np.savez(PATH + "ReplayMemory", save_states, save_actions, save_rewards, save_new_states, save_terminals, save_TD_errors)



if TRAIN:
    train()

