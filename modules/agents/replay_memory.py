import os
import random
import numpy as np
import torch
class DQNMemory:
    def __init__(self, args):
        self.args = args
        self.entry_size = self.args.DQN_input_size
        self.memory_size = self.args.DQN_memory_size
        if self.args.isSharePara:
            self.actions = np.zeros([self.memory_size, 1,self.args.num_dqn], dtype=np.uint8)
            self.tot_rewards = np.zeros([self.memory_size, 1,self.args.num_dqn], dtype=np.float64)
            self.ind_rewards = np.zeros([self.memory_size, 1,self.args.num_dqn], dtype=np.float64)
            self.prestate = np.zeros([self.memory_size, self.entry_size,self.args.num_dqn])
            self.poststate = np.empty([self.memory_size, self.entry_size,self.args.num_dqn])
            self.s_t = np.empty([self.memory_size, self.args.s_t_input_size,self.args.num_dqn])
            self.s_t_plus_1 = np.empty([self.memory_size, self.args.s_t_input_size,self.args.num_dqn])
        else:
            self.actions = np.zeros([self.memory_size,1], dtype=np.uint8)
            self.tot_rewards = np.zeros([self.memory_size,1], dtype=np.float64)
            self.ind_rewards = np.zeros([self.memory_size, 1], dtype=np.float64)
            self.prestate = np.zeros([self.memory_size, self.entry_size])
            self.poststate = np.empty([self.memory_size, self.entry_size])
            self.s_t = np.empty([self.memory_size, self.args.s_t_input_size])
            self.s_t_plus_1 = np.empty([self.memory_size, self.args.s_t_input_size])
        self.batch_size = self.args.DQN_batch_size
        self.count = 0
        self.current = 0

    def add(self, prestate, poststate, ind_reward, tot_reward, action, s_t, s_t_plus_1):
        self.actions[self.current] = action
        self.tot_rewards[self.current] = tot_reward
        self.ind_rewards[self.current] = ind_reward
        self.prestate[self.current] = prestate
        self.poststate[self.current] = poststate
        self.s_t[self.current] = s_t
        self.s_t_plus_1[self.current] = s_t_plus_1
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self, indexes):
        # if self.count < self.batch_size:
        #     indexes = range(0, self.count)
        # else:
        #     indexes = random.sample(range(0, self.count), self.batch_size)
        prestate = self.prestate[indexes]
        poststate = self.poststate[indexes]
        # poststate = self.prestate[np.array(indexes)+1]
        actions = self.actions[indexes]
        tot_rewards = self.tot_rewards[indexes]
        ind_rewards = self.ind_rewards[indexes]
        s_t = self.s_t[indexes]
        s_t_plus_1 = self.s_t_plus_1[indexes]
        # s_t_plus_1 = self.s_t[np.array(indexes)+1]
        return prestate, poststate, actions, tot_rewards, ind_rewards, s_t, s_t_plus_1

class PPOMemory:
    def __init__(self, args):
        self.args = args
        self.count = 0
        self.data_buffer = []  # To store the experience
        # the number of experience tuple in data_buffer

    def sample(self):  # Sample all the data
        l_s, l_a, l_a_p, l_r_tot, l_r_ind, l_s_, l_s_t, l_s_t_plus_1 = [], [], [], [], [], [], [], []
        for item in self.data_buffer:
            s, a, a_prob, r_tot, r_ind, s_, s_t, s_t_plus_1 = item
            l_s.append(torch.tensor(np.array([s]), dtype=torch.float))
            l_a.append(torch.tensor([[a]], dtype=torch.long))
            l_a_p.append(torch.tensor([a_prob], dtype=torch.float))
            l_r_tot.append(torch.tensor([r_tot], dtype=torch.float))
            l_r_ind.append(torch.tensor([r_ind], dtype=torch.float))
            l_s_.append(torch.tensor(np.array([s_]), dtype=torch.float))
            l_s_t.append(torch.tensor(np.array([s_t]), dtype=torch.float))
            l_s_t_plus_1.append(torch.tensor(np.array([s_t_plus_1]), dtype=torch.float))
        s = torch.cat(l_s, dim=0).to(self.args.device)
        a = torch.cat(l_a, dim=0).to(self.args.device)
        a_prob = torch.cat(l_a_p, dim=0).unsqueeze(1).to(self.args.device)
        r_tot = torch.cat(l_r_tot, dim=0).unsqueeze(1).to(self.args.device)
        r_ind = torch.cat(l_r_ind, dim=0).unsqueeze(1).to(self.args.device)
        s_ = torch.cat(l_s, dim=0).squeeze(1).to(self.args.device)
        s_t = torch.cat(l_s_t, dim=0).squeeze(1).to(self.args.device)
        s_t_plus_1 = torch.cat(l_s_t, dim=0).squeeze(1).to(self.args.device)
        self.data_buffer = []
        self.count = 0
        return s,  s_, a, r_tot, r_ind, s_t, s_t_plus_1, a_prob

    def add(self, transition):
        self.data_buffer.append(transition)
        self.count = self.count + 1

