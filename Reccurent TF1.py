import imageio
import random
import gym
import tensorflow as tf
import numpy as np
from array2gif import write_gif

TRAIN = True

ENV_NAME = 'Pong-v0'
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

        self.input = tf.placeholder(shape=[None, self.frame_height, self.frame_width, 1], dtype=tf.float32)
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
            inputs=self.conv3, filters=1024, kernel_size=[7, 7], strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')

        # Splitting into value and advantage stream
        # value,  num_of_splits, axis
        self.flat = tf.layers.flatten(self.conv4)
        self.lstm = tf.keras.layers.LSTM(units=hidden, input_shape=(1024,) , return_sequences=True)(
            tf.expand_dims(self.flat, 0))
        self.valuestream, self.advantagestream = tf.split(tf.reshape(self.lstm, [hidden, 1]), 2)

        self.advantage = tf.layers.dense(
            inputs=self.advantagestream, units=self.n_actions,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name="advantage")

        self.value = tf.layers.dense(
            inputs=self.valuestream, units=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value')

        # Combining value and advantage into Q-values as described above
        self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
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


class ReplayMemory(object):
    """Replay Memory that stores the last size=1,000,000 transitions"""

    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width, 1), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def load(self, file):
        self.actions = file['arr_0']
        self.rewards = file['arr_1']
        self.frames = file['arr_2']
        self.terminal_flags = file['arr_3']
        self.count = len(self.actions)
        self.current = self.count - 1


    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width, 1):
            raise ValueError('Dimension of frame is wrong!\n Expected ('+str(self.frame_height) +','+ str(self.frame_width) +', 1)\n Got '+str(frame.shape))

        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        return np.transpose(self.frames, axes=(1, 2, 0)), self.actions[self.indices], self.rewards[
            self.indices], np.transpose(self.new_states, axes=(1, 2, 0)), self.terminal_flags[self.indices]

    def get_random_ep(self):
        idx = random.randrange(0, self.count)
        l_idx = idx - 1
        r_idx = idx

        while l_idx > 0 and self.terminal_flags[l_idx] != False:
            l_idx -= 1

        while r_idx < self.count and self.terminal_flags[r_idx] != False:
            r_idx += 1

        states     = self.frames[l_idx+1:r_idx]
        new_states = self.frames[l_idx+2:r_idx+1]

        return np.transpose(states, axes=(1, 2, 0)),
        self.actions[l_idx+1:r_idx+1],
        self.rewards[l_idx+1:r_idx+1],
        np.transpose(new_states, axes=(1, 2, 0)),
        self.terminal_flags[l_idx+1:r_idx+1]

def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
    """
    Args:
        session: A tensorflow sesson object
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
    states, actions, rewards, new_states, terminal_flags = replay_memory.get_random_ep()
    # The main network estimates which action is best (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    # The target network estimates the Q-values (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(batch_size), arg_q_max]

    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q * (1-terminal_flags))
    # Gradient descend step to update the parameters of the main network
    loss, _, TD_error = session.run([main_dqn.loss, main_dqn.update, main_dqn.TD_Error],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions})

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
        processed_frame = self.process_frame(sess, frame)  # (★★★)
        self.state = processed_frame

        return terminal_life_lost

    def step(self, sess, action):
        """
        Args:
            sess: A Tensorflow session object
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)  # (5★)

        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']

        self.state = self.process_frame(sess, new_frame)

        if self.state.shape != (84, 84, 1):
            raise ValueError('Dimension of frame is wrong!\n Expected (84, 84, 1)\n Got '+str(self.state.shape))

        return self.state, reward, terminal, terminal_life_lost, new_frame


def clip_reward(reward):
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1


tf.reset_default_graph()

# Control parameters
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
EVAL_FREQUENCY = 100000          # Number of frames the agent sees between evaluations
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
MEMORY_SIZE = 1000000            # Number of transitions stored in the replay memory
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


atari = Atari(ENV_NAME, NO_OP_STEPS)

print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                atari.env.unwrapped.get_action_meanings()))
# main DQN and target DQN networks:
with tf.variable_scope('mainDQN'):
    MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE)   # (★★)
with tf.variable_scope('targetDQN'):
    TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)               # (★★)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

def save_vid(images, fname):
    imageio.mimsave(fname, images)

def train():
    """Contains the training and evaluation loops"""
    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS)  # (★)
    update_networks = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)

    explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n,
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
        max_frames=MAX_FRAMES)


    with tf.Session() as sess:
        sess.run(init)
        #saver.restore(sess, "/home/Kapok/BaseLines_Saves/Breakout/Weights/9866587")
        #my_replay_memory.load(np.load("/home/Kapok/BaseLines_Saves/Breakout/Memory.npz"))
        frame_number = 0
        run = 0
        rewards = []
        log_list = [[],[]]
        gif=[]
        processed_gif = []
        deadFrames=0
        terminal = False
        is_eval =False
        while frame_number < MAX_FRAMES:

            epoch_frame = 0
            is_eval = True
            while epoch_frame < EVAL_FREQUENCY:
                terminal_life_lost = atari.reset(sess)
                episode_reward_sum = 0
                run += 1
                for _ in range(MAX_EPISODE_LENGTH):

                    # while run == 1 and not terminal:
                    #     processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, random.randrange(4))
                    #     gif.append(np.transpose(new_frame, axes=(1, 0, 2)))
                    #     processed_gif.append(processed_new_frame)
                    #     deadFrames+=1
                    #     if deadFrames%100==0:
                    #         print(deadFrames)
                    #         print(new_frame.shape)
                    # if run == 1:
                    #     write_gif(gif, "/home/kapok/Gifs/DeadFrames"+str(run)+".gif", fps=30)
                    #     save_vid(processed_gif, "/home/kapok/Gifs/DeadFramesProcessed"+str(run)+".gif")
                    #
                    #     print("finally out after",deadFrames,"dead frames")

                    state = atari.state
                    action = explore_exploit_sched.get_action(sess, frame_number, state, evaluation=is_eval)

                    processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward

                    if is_eval:
                        gif.append(np.transpose(new_frame, axes=(1, 0, 2)))
                        processed_gif.append(processed_new_frame)

                    # Clip the reward
                    clipped_reward = clip_reward(reward)

                    # (7★) Store transition in the replay memory
                    my_replay_memory.add_experience(action=action,
                                                    frame=state,
                                                    reward=clipped_reward,
                                                    terminal=terminal_life_lost)



                    if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        update_networks(sess)  # (9★)

                    if terminal:
                        if is_eval:
                            print("\n############ EVALUATION ############\n")
                            is_eval = False
                            write_gif(gif, "/home/kapok/Gifs/DeadFrames_" + str(run) + ".gif", fps=30)
                            save_vid(processed_gif, "/home/kapok/Gifs/DeadFramesProcessed_"+str(run)+".gif")
                            gif, processed_gif = [], []
                        print("Run: " + str(run) + "  Reward: " + str(episode_reward_sum) + "  Explore Rate: " + str(explore_exploit_sched.get_epsilon(frame_number)) + "  Frame Count: " + str(frame_number))
                        terminal = False

                        if frame_number > REPLAY_MEMORY_START_SIZE:
                            loss, TD_error = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN, BS,
                                                   gamma=DISCOUNT_FACTOR)
                            log_list[0].append(loss)
                            log_list[1].append(TD_error)
                        break

            # Save the network parameters, Memory & Logs
            saver.save(sess, "/data/Saves/Recurrent/Pong/Weights/" + str(frame_number))
            np.savez("/data/Saves/Recurrent/Pong/Memory/Memory", my_replay_memory.actions,
                     my_replay_memory.rewards, my_replay_memory.frames, my_replay_memory.terminal_flags)
            np.savez("/data/Saves/Recurrent/Pong/Logs/Logs_" + str(frame_number), log_list[0], log_list[1])
            log_list = [[], []]
            print("saved")



if TRAIN:
    train()
