import unittest
import random
import time
import numpy as np


class Reccurent_TEST(unittest.TestCase):
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
                raise ValueError('Dimension of frame is wrong!\n Expected (' + str(self.frame_height) + ',' + str(
                    self.frame_width) + ', 1)\n Got ' + str(frame.shape))

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

            while l_idx >= 0 and self.terminal_flags[l_idx] != True:
                l_idx -= 1

            while r_idx < self.count and self.terminal_flags[r_idx] != True:
                r_idx += 1

            states = self.frames[l_idx + 1:r_idx]
            new_states = self.frames[l_idx + 2:r_idx + 1]

            return states,\
                   self.actions[l_idx + 1:r_idx],\
                   self.rewards[l_idx + 1:r_idx],\
                   new_states,\
                   self.terminal_flags[l_idx + 1:r_idx]

    def setUp(self):
        pass

    def test_get_ep_return_size(self):
        my_replay_memory = self.ReplayMemory()
        for i in range(10):
            my_replay_memory.add_experience(0, np.ones([84, 84, 1]), 0, False)
        my_replay_memory.add_experience(0, np.ones([84, 84, 1]), 0, True)

        for i in range(10):
            my_replay_memory.add_experience(0, np.ones([84, 84, 1]), 0, False)
        my_replay_memory.add_experience(0, np.ones([84, 84, 1]), 0, True)

        s,a,r,ns,t = my_replay_memory.get_random_ep()

        self.assertTupleEqual(s.shape, ns.shape)
        self.assertEquals(len(ns), len(s))
        self.assertEquals(len(ns), 10)

    def test_states_match_info(self):
        my_replay_memory = self.ReplayMemory()
        for i in range(10):
            my_replay_memory.add_experience(0, np.ones([84, 84, 1]), 0, False)
        my_replay_memory.add_experience(0, np.ones([84, 84, 1]), 0, True)

        s,a,r,ns,t = my_replay_memory.get_random_ep()

        self.assertEquals(len(ns), len(s))
        self.assertEquals(len(s),len(a))
        self.assertEquals(len(t),len(s))
        self.assertEquals(len(r),len(s))
        self.assertFalse(t[-1])

if __name__ == '__main__':
    unittest.main()
