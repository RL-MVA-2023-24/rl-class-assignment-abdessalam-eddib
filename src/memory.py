import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


def fill_buffer(env, agent, buffer_size):
    state, _ = env.reset()
    for _ in range(buffer_size):
        action = agent.greedy_action(state)
        next_state, reward, done, trunc, _ = env.step(action)
        agent.memory.append(state, action, reward, next_state, done)
        if done or trunc:
            state, _ = env.reset()
        else:
            state = next_state
