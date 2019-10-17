import random
import gym
import numpy as np
from collections import deque
import tflearn
import tensorflow as tf
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

ENV_NAME = "BreakoutDeterministic-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
IMG_WIDTH = 72
IMG_HEIGHT = 92

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        tf.reset_default_graph()

        network = tflearn.input_data(shape=[None, IMG_HEIGHT, IMG_WIDTH, 1], name='input')

        network = tflearn.conv_2d(network, 16, [8, 8], strides=[4, 4], activation='relu', regularizer="L2")
        network = tflearn.conv_2d(network, 32, [5, 4], strides=[2, 2], activation='relu', regularizer="L2")

        network = tflearn.flatten(network)
        network = tflearn.fully_connected(network, 256, activation='relu')
        network = tflearn.fully_connected(network, action_space, activation='linear')


        network = tflearn.regression(network, optimizer='adam', learning_rate=LEARNING_RATE, loss='categorical_crossentropy',
                             name='targets')

        self.model = tflearn.DNN(network, tensorboard_dir='log')



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                predictions = self.model.predict(state_next)
                q_update = (reward + GAMMA * np.amax(predictions))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def preprocess(state):
    # Turn to grey
    state = rgb2gray(state)

    # Crop 26px from top & 16px from bottom
    state = state[26:-16]

    # Downscale 184, 144 
    state = resize(state, [IMG_HEIGHT, IMG_WIDTH], anti_aliasing=False)

    # Reshape
    state = np.array(state).reshape([-1, IMG_HEIGHT, IMG_WIDTH, 1])
    return state


def main_loop():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    allOutput = ""
    while run < 1000:

        run += 1
        state = env.reset()
        state = preprocess(state)

        step = 0
        while True:
            step += 1
            # env.render()
            # plt.imshow(state, cmap="gray")
            # plt.show()

            action = dqn_solver.act(state)

            state_next, reward, terminal, info = env.step(action)
            state_next = preprocess(state_next)

            reward = reward if not terminal else -reward

            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            
            if terminal:
                output = "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step)
                allOutput += output+"\n"
                print(output)
                break
        dqn_solver.experience_replay()
    return allOutput


if __name__ == "__main__":
    out = main_loop()
    
print(out)
