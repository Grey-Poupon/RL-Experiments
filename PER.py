import pickle
import random
import gym
import tensorflow as tf
import numpy as np
from PER_Memory import Item, SumTree, PEReplayMemory
import time
import matplotlib.pyplot as plt

TRAIN = True

#ENV_NAME = 'BreakoutDeterministic-v4'
ENV_NAME = 'Qbert-v0'
#ENV_NAME = 'PongDeterministic-v4'
# You can increase the learning rate to 0.00025 in Pong for quicker results

class FrameProcessor(object):
    """Resizes and converts RGB Atari frames to grayscale"""

    def __init__(self, frame_height=84, frame_width=84):
        """
        Args:
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
        self.processed = tf.image.resize_images(self.processed,
                                                [self.frame_height, self.frame_width],
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def __call__(self, session, frame):
        """
        Args:
            session: A Tensorflow session object
            frame: A (210, 160, 3) frame of an Atari game in RGB
        Returns:
            A processed (84, 84, 1) frame in grayscale
        """
        return session.run(self.processed, feed_dict={self.frame: frame})


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

        self.input = tf.placeholder(shape=[None, self.frame_height,
                                           self.frame_width, self.agent_history_length],
                                    dtype=tf.float32)
        # Normalizing the input
        self.inputscaled = self.input / 255

        # Convolutional layers
        self.conv1 = tf.layers.conv2d(
            inputs=self.inputscaled, filters=32, kernel_size=[8, 8], strides=4,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=2,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3')
        self.conv4 = tf.layers.conv2d(
            inputs=self.conv3, filters=hidden, kernel_size=[7, 7], strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')

        # Splitting into value and advantage stream
        self.valuestream, self.advantagestream = tf.split(self.conv4, 2, 3)
        self.valuestream = tf.layers.flatten(self.valuestream)
        self.advantagestream = tf.layers.flatten(self.advantagestream)
        self.advantage = tf.layers.dense(
            inputs=self.advantagestream, units=self.n_actions,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name="advantage")
        self.value = tf.layers.dense(
            inputs=self.valuestream, units=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value')

        # Combining value and advantage into Q-values as described above
        self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.best_action = tf.argmax(self.q_values, 1)

        # The next lines perform the parameter update. This will be explained in detail later.

        # targetQ according to Bellman equation:
        # Q = r + gamma*max Q', calculated in the function learn()
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # Action that was performed
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        # Q value of the action that was performed
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                               axis=1)
        self.TD_Error = self.target_q-tf.reduce_max(self.q_values, axis=1)

        # Parameter updates
        self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_q, predictions=self.Q))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

        # weight = (N·P(j))−β / Max Weight
        self.N = tf.placeholder(dtype=tf.float32)
        self.P = tf.placeholder(dtype=tf.float32)
        self.M = tf.placeholder(dtype=tf.float32)
        self.B = tf.placeholder(dtype=tf.float32)

        # Calculcate weight as per PER
        self.importance_weight = tf.divide(tf.pow((self.N * self.P), -self.B), self.M)


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
            eps = -1
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2 * frame_number + self.intercept_2

        return eps

    def get_action(self, session, frame_number, state, evaluation=False):
        """
        Args:
            session: A tensorflow session object
            frame_number: Integer, number of the current frame
            state: A (84, 84, 4) sequence of frames of an Atari game in grayscale
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions - 1 determining the action the agent perfoms next
        """

        if np.random.rand(1) < self.get_epsilon(frame_number, evaluation):
            return np.random.randint(0, self.n_actions)
        return session.run(self.DQN.best_action, feed_dict={self.DQN.input: [state]})[0]


def learn(session, PER_memory, main_dqn, target_dqn, batch_size, gamma, beta=1):
    """
    Args:
        session: A tensorflow sesson object
        PER_memory: A ReplayMemory object
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
    states, actions, rewards, new_states, terminal_flags, probabilities, items = PER_memory.get_minibatch()

    # The main network estimates which action is best (in the next
    # state s', new_states is passed!)
    # for every transition in the minibatch

    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    # The target network estimates the Q-values (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})

    double_q = q_vals[range(batch_size), arg_q_max]
    # We use an importance sampling weight to reduce the bias introduces using PER

    importance_sampling_weight = session.run(main_dqn.importance_weight, feed_dict={
                                                        main_dqn.N: PER_memory.tree.get_num_leaves(),
                                                        main_dqn.P: probabilities,
                                                        main_dqn.M: PER_memory.tree.get_max_weight(beta),
                                                        main_dqn.B: beta})
    #print(importance_sampling_weight)

    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = (rewards + gamma*double_q * (1-terminal_flags)) * importance_sampling_weight
    # Gradient descend step to update the parameters of the main network
    loss, _, TD_error = session.run([main_dqn.loss, main_dqn.update, main_dqn.TD_Error],
                                    feed_dict={main_dqn.input: states,
                                               main_dqn.target_q: target_q,
                                               main_dqn.action: actions})
    for i, error in enumerate(TD_error):
        items[i].TD_error = error

    return loss, TD_error


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
        update_ops = []
        for i, var in enumerate(self.main_dqn_vars):
            copy_op = self.target_dqn_vars[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops

    def __call__(self, sess):
        """
        Args:
            sess: A Tensorflow session object
        Assigns the values of the parameters of the main network to the
        parameters of the target network
        """
        update_ops = self._update_target_vars()
        for copy_op in update_ops:
            sess.run(copy_op)


class Atari(object):
    """Wrapper for the environment provided by gym"""

    def __init__(self, envName, no_op_steps=10, agent_history_length=4):
        self.env = gym.make(envName)
        self.process_frame = FrameProcessor()
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length

    def reset(self, sess, evaluation=False):
        """
        Args:
            sess: A Tensorflow session object
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
        processed_frame = self.process_frame(sess, frame)
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)

        return terminal_life_lost

    def step(self, sess, action):
        """
        Args:
            sess: A Tensorflow session object
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)

        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_new_frame = self.process_frame(sess, new_frame)

        self.state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2)

        return processed_new_frame, reward, terminal, terminal_life_lost, new_frame


def clip_reward(reward):
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1


tf.reset_default_graph()

# Control parameters
MAX_EPISODE_LENGTH = 1000       # Equivalent of 5 minutes of gameplay at 60 frames per second
EVAL_FREQUENCY = 250000          # Number of frames the agent sees between evaluations
EVAL_STEPS = 10000               # Number of frames for one evaluation
NETW_UPDATE_FREQ = 10000         # Number of chosen actions between updating the target network.
                                 # According to Mnih et al. 2015 this is measured in the number of
                                 # parameter updates (every four actions), however, in the
                                 # DeepMind code, it is clearly measured in the number
                                 # of actions the agent choses
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions,
                                 # before the agent starts learning
MAX_FRAMES = 10000000            # Total number of frames the agent sees
MEMORY_SIZE = 400000            # Number of transitions stored in the replay memory
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

PATH = "output/"                 # Gifs and checkpoints will be saved here
SUMMARIES = "summaries"          # logdir for tensorboard
RUNID = 'run_1'
SAVE_SWITCH = False

atari = Atari(ENV_NAME, NO_OP_STEPS)

print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                atari.env.unwrapped.get_action_meanings()))
# main DQN and target DQN networks:
with tf.variable_scope('mainDQN'):
    MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE)
with tf.variable_scope('targetDQN'):
    TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')


def save_data(frame_number, my_replay_memory, log_list, sess,SAVE_SWITCH, test=False):

    saver.save(sess, "/home/Kapok/Saves/PER/Qbert/Saves/" + str(frame_number))
#    with open('memory.pkl', 'wb') as output:
#        pickle.dump(my_replay_memory, output, pickle.HIGHEST_PROTOCOL)
    fname = "tree_A.pkl" if SAVE_SWITCH else "tree_B.pkl"
    SAVE_SWITCH = not SAVE_SWITCH
    with open(fname, 'wb') as output:
        pickle.dump(my_replay_memory.tree.tree, output, pickle.HIGHEST_PROTOCOL)
    np.save("/home/Kapok/Saves/PER/Qbert/Logs/Logs_" + str(frame_number), log_list)

def load_data(sess, test=False):
    saver.restore(sess, "/home/Kapok/Saves/PER/Breakout/Saves/1000420")
    with open('tree.pkl', 'rb') as input:
        tree = pickle.load(input)
        print("Loaded Memory")
        return sess, tree



def train():
    """Contains the training and evaluation loops"""
    my_replay_memory = PEReplayMemory(size=MEMORY_SIZE, batch_size=BS)
    update_networks = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)

    explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n,
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
        max_frames=MAX_FRAMES)


    with tf.Session() as sess:
        sess.run(init)
       # sess, tree = load_data(sess)
       # my_replay_memory.load_tree(tree)
        frame_number = 0
        run = 0
        epoch_run = 0
        rewards = []
        log_list = []
        is_eval =False
        print("Start\n\n")
        while frame_number < MAX_FRAMES:

            epoch_frame = 0
            epoch_run = 0
            is_eval = True
            while epoch_frame < EVAL_FREQUENCY:

                start = time.perf_counter()
                terminal_life_lost = atari.reset(sess)
                episode_reward_sum = 0
                run += 1
                epoch_run += 1
                for f in range(MAX_EPISODE_LENGTH):

                    state = atari.state
                    action = explore_exploit_sched.get_action(sess, frame_number, state, evaluation=is_eval)
                    if f > 150 and epoch_run == 1:
                         action = 1
                         print(action)
                    processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)

                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward                                           

                    # Clip the reward
                    clipped_reward = clip_reward(reward)

                    # Store transition in the replay memory
                    my_replay_memory.add_experience(action=action,
                                                    state=state,
                                                    new_state=processed_new_frame,
                                                    reward=clipped_reward,
                                                    terminal=terminal_life_lost)

                    if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        loss, TD_error = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN,
                                     BS, gamma=DISCOUNT_FACTOR)
                        log_list.append([loss, TD_error])
                    if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        update_networks(sess)

                    if terminal:
                        end = time.perf_counter()
                        if is_eval:
                            print("\n############ EVALUATION ############\n")
                            is_eval = False
                        print("Run: " + str(run) + "  Reward: " + str(episode_reward_sum) + "  Explore Rate: " + str(explore_exploit_sched.get_epsilon(frame_number)) + "  Frame Count: " + str(frame_number) + "  Time:"+str(end-start))
                        terminal = False
                        break


            # Save the network parameters, Memory & Logs
            save_data(frame_number, my_replay_memory, log_list, sess, SAVE_SWITCH)
            print("saved")



if TRAIN:
    train()

