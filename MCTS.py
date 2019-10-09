import random
import gym
import numpy as np
from collections import deque
import tflearn
from copy import deepcopy


class env_node(object):
    def __init__(self, env, parent=None):
        if parent is None:
            self.is_root = True
            self.root = self
        else:
            self.is_root = False
            self.root = parent.root

        self.parent = parent
        self.env = deepcopy(env)
        self.children = []
        self.is_done = False

        self.visit_count = 0
        self.win_ratio = 0


    def make_children(self):
        child1 = deepcopy(self.env)
        child2 = deepcopy(self.env)

        child1.step(0)
        child2.step(1)

        self.children = [env_node(child1, self), env_node(child2, self)]

    def rollout(self, model):

        env = deepcopy(self.env)

        total_return = 0
        # Cart_Position , Cart_Velocity, Pole_Angle ,Tip_Velocity
        observation = np.array(env.state)

        while True:

            # Pick Action
            action = np.argmax(model.predict([observation]))
            # Do Action
            observation, reward, done, _ = env.step(action)

            # Count reward
            total_return += reward

            if done:
                if total_return == 0:
                    self.set_done()
                return total_return

    def calculate_UCT(self):
        return self.win_ratio + np.sqrt(2) * np.sqrt(np.log(self.root.visit_count)/ self.visit_count)

    def backpropagate(self, model):
        reward = self.rollout(model)
        win = reward/100

        node = self

        while node is not None:
            node.win_ratio += (win - node.win_ratio)/node.visit_count
            node = node.parent



    def best_child(self):
        if self.is_done:
            return None
        if self.children[0].is_done:
            return self.children[1]
        if self.children[1].is_done:
            return self.children[0]

        if self.children[0].calculate_UCT() >= self.children[1].calculate_UCT():
             return self.children[0]
        else:
            return self.children[1]

    def has_children(self):
        return len(self.children) > 0

    def get_action_scores(self):
        return [self.children[0].calculate_UCT(), self.children[1].calculate_UCT()]

    def set_done(self):
        self.is_done = True
        if self.parent is not None and self.parent.children[0].is_done and self.parent.children[1].is_done:
            self.parent.set_done()

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.win_ratio) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

def neural_network_model(input_size, LR):
    network = tflearn.input_data(shape=[None, input_size], name = 'input')

    network = tflearn.fully_connected(network, 24, activation='relu')
    network = tflearn.dropout(network, 0.8)

    network = tflearn.fully_connected(network, 24, activation='relu')
    network = tflearn.dropout(network, 0.8)

    network = tflearn.fully_connected(network, 2, activation='softmax')
    network = tflearn.regression(network, optimizer='adam', learning_rate=LR, loss = 'categorical_crossentropy', name = 'targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def flatten_tree(node, array=[]):
    array.append(node)
    if node.has_children():
        array = flatten_tree(node.children[0], array)
        array = flatten_tree(node.children[1], array)
    return array


def cartpole(model):
    env = gym.make("CartPole-v1")
    for i in range(20):
        env.reset()
        root = env_node(env)
        root.make_children()

        curr_node = root

        rollouts = 0

        while rollouts < 100:
     #       print(root)
            curr_node.visit_count += 1
            if curr_node.has_children():
                curr_node = curr_node.best_child()
                if curr_node is None:
                    break
            else:
                if curr_node.visit_count == 1:
                    curr_node.backpropagate(model)
                    curr_node = root
                    rollouts += 1
                    #print(rollouts)
                else:
                    curr_node.make_children()
                    curr_node = curr_node.children[0]

        nodes = flatten_tree(root)
        transistions = []

        for node in nodes:
            if node.has_children():
                t = [node.env.state]
                t.extend(node.get_action_scores())
                transistions.append(t)

        for values in transistions:
            model.fit([values[0]], [values[1:]], n_epoch=i)

        model.save("MCTS_Model")

        reward = root.rollout(model)
        print("Trained model Score = " + str(reward))
    print(root)




np.random.seed(123)
cartpole(neural_network_model(4, 0.001))












