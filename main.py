import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from collections import Counter


def neural_network_model(input_size, LR):
    network = input_data(shape=[None, input_size], name = 'input')

    network = fully_connected(network, 24, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 24, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss = 'categorical_crossentropy', name = 'targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def wrap_array(arr, n=2):
    arr = np.array(arr)
    while len(arr.shape) < n:
        arr = np.array([arr])
    return arr


def model_picks_action(model, observation):

    observation = wrap_array(observation)
    values = model.predict(observation)

    # Return best Action, Expected Return for this Action
    idx = np.argmax(values[0])
    return idx, values[0][idx]


def train_model_1_step(model, transition):
    discount_value = 0.95
    State, ActionIdx, Reward, State2, done = transition

    State = wrap_array(State)
    values = model.predict(State)

    State2 = wrap_array(State2)

    if not done:
        successor_values = model.predict(State2)
        max_successor_value = np.max(successor_values)
        values[0][ActionIdx] = Reward + discount_value * max_successor_value
    else:
        values[0][ActionIdx] = -1

    #print("\t\tAction " + str(ActionIdx) + "  Valued at " + str(values[0][ActionIdx]) + "  Reward for A was "+str(Reward))
    #print("\t\tMove towards "+str(Reward + discount_value * max_successor_value))

    model.fit(State, values, n_epoch=5)


def play_n_games(model, n=5, name='LatestModel'):

    env = gym.make("CartPole-v0")
    env.reset()

    explore_max = 1.0
    explore_min = 0.01
    explore_decay = 0.999
    explore_rate = explore_max

    all_transitions = []
    all_returns = []

    for ep in range(n):
        total_return = 0
        rand_moves = 0
        env.reset()
        done = False
        while not done:

            # observe State # Cart_Position , Cart_Velocity, Pole_Angle ,Tip_Velocity
            observation = np.array(env.state)

            # Pick Action or Explore
            if np.random.rand() < explore_rate:
                action = env.action_space.sample()
                rand_moves += 1
            else:
                action, _ = model_picks_action(model, observation)


            #print("\nModel chooses Action "+str(action))

            # Take Action, receive Reward, transition to New State
            observation2, reward, done, _ = env.step(action)

            # Update memory with S A R S`
            all_transitions.append([observation, action, reward, observation2, done])
            np.random.shuffle(all_transitions)
            train_model_1_step(model, all_transitions.pop())

            total_return += reward
            explore_rate *= explore_decay
            explore_rate = explore_rate if explore_rate > explore_min else explore_min

            if done:
                print("EP "+str(ep+1)+"\tTotal Return = "+str(total_return)+"\t Explore Moves = "+str(rand_moves)+"\t epsilon = "+str(explore_rate))
                all_returns.append(total_return)
                break

    return model, all_returns


model = neural_network_model(4, 0.001)
model, _ = play_n_games(model, 500)


