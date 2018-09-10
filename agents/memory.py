import random
from collections import namedtuple, deque
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "prev_reward"])

    def add(self, state, action, reward, next_state, done, prev_reward):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, prev_reward)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if len(self.memory) > self.batch_size:
            return random.sample(self.memory, k=self.batch_size)
        else:
            return self.memory

    def sample_all(self):
        return self.memory

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# code adapted from https://gist.github.com/monkut/2e8bec49b0659d941abc
class RunningVariance(object):
    def __init__(self, shape=(), epsilon=1e-3):
        self.epsilon = epsilon
        self.count = 0
        self.mean = np.zeros(shape)
        self.last_mean = np.zeros(shape)
        self.sq_distance = np.zeros(shape)  # square distance from the mean
        self.last_sq_distance = np.zeros(shape)
        self.stddev = np.ones(shape)

    def update(self, value):
        val = np.array(value)
        self.count += 1
        self.mean = self.last_mean + (val - self.last_mean) / self.count
        self.sq_distance = self.last_sq_distance + (val - self.last_mean) * (val - self.mean)
        if self.count > 1:
            self.stddev = np.sqrt(self.sq_distance / (self.count - 1))
            self.stddev[self.stddev == 0] = self.epsilon
        self.last_mean = self.mean
        self.last_sq_distance = self.sq_distance

    def normalize(self, value):
        val = np.array(value)
        return (val - self.mean) / self.stddev

    def denormalize(self, value):
        val = np.array(value)
        return val * self.stddev + self.mean
