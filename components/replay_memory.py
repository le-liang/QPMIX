import os
import random
import numpy as np
from config import config
cfg = config()
import torch
class DQNMemory:
    def __init__(self):
        self.entry_size = cfg.DQN_input_size
        self.memory_size = cfg.DQN_memory_size
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.float64)
        self.prestate = np.empty((self.memory_size, self.entry_size), dtype=np.float16)
        self.poststate = np.empty((self.memory_size, self.entry_size), dtype=np.float16)
        self.batch_size = cfg.DQN_batch_size
        self.count = 0
        self.current = 0

    def add(self, prestate, poststate, reward, action):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.prestate[self.current] = prestate
        self.poststate[self.current] = poststate
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self):

        if self.count < self.batch_size:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0, self.count), self.batch_size)
        prestate = self.prestate[indexes]
        poststate = self.poststate[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        return prestate, poststate, actions, rewards


class PPOMemory:
    def __init__(self):
        self.counter = 0
        self.data_buffer = []  # To store the experience
        # the number of experience tuple in data_buffer

    def sample(self):  # Sample all the data
        l_s, l_a, l_a_p, l_r, l_s_ = [], [], [], [], []
        for item in self.data_buffer:
            s, a, a_prob, r, s_ = item
            l_s.append(torch.tensor(np.array([s]), dtype=torch.float))
            l_a.append(torch.tensor([[a]], dtype=torch.long))
            l_a_p.append(torch.tensor([a_prob], dtype=torch.float))
            l_r.append(torch.tensor([r], dtype=torch.float))
            l_s_.append(torch.tensor(np.array([[s_]]), dtype=torch.float))
        s = torch.cat(l_s, dim=0).to(cfg.device)
        a = torch.cat(l_a, dim=0).to(cfg.device)
        a_prob = torch.cat(l_a_p, dim=0).unsqueeze(1).to(cfg.device)
        r = torch.cat(l_r, dim=0).unsqueeze(1).to(cfg.device)
        s_ = torch.cat(l_s_, dim=0).squeeze(1).to(cfg.device)
        self.data_buffer = []
        return s, a, a_prob, r, s_

    def add(self, transition):
        self.data_buffer.append(transition)
        self.counter = self.counter + 1

