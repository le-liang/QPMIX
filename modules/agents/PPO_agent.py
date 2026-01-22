import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from os.path import dirname
from modules.agents.replay_memory import PPOMemory
from modules.agents.replay_memory import DQNMemory
import random
from config import config
import torch
args = config(False)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

class Actor(nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, n_hidden3, output_size):
        super(Actor, self).__init__()
        self.fc_1 = nn.Linear(input_size, n_hidden1)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(n_hidden2, n_hidden3)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(n_hidden3, output_size)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        action_prob = F.softmax(self.fc_4(x), dim=1)
        return action_prob
class DQN(nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, n_hidden3, output_size):
        super(DQN, self).__init__()
        self.fc_1 = nn.Linear(input_size, n_hidden1)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(n_hidden2, n_hidden3)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(n_hidden3, output_size)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))
        # x = F.relu(self.fc_4(x))
        x = F.leaky_relu(self.fc_4(x))
        return x

class Critic(nn.Module):
    def __init__(self,input_size, n_hidden1, n_hidden2, n_hidden3):
        super(Critic, self).__init__()
        self.fc_1 = nn.Linear(input_size, n_hidden1)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(n_hidden2, n_hidden3)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(n_hidden3, 1)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = self.fc_4(x)
        return x

class PPO_Agent:
    def __init__(self, args):
        self.args = args
        self.agent_type = 'PPO'
        self.LAMBDA = 0.95
        self.discount = 0.99
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.K_epoch = 8

        self.actor = Actor(self.args.PPO_input_size, self.args.PPO_hidden_1, self.args.PPO_hidden_2, self.args.PPO_hidden_3, self.args.PPO_output_size).to(self.args.device)
        self.old_actor = Actor(self.args.PPO_input_size, self.args.PPO_hidden_1, self.args.PPO_hidden_2, self.args.PPO_hidden_3, self.args.PPO_output_size).to(self.args.device)  # Old policy network
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.PPO_actor_lr)
        self.model = DQN(self.args.DQN_input_size, self.args.DQN_hidden_1, self.args.DQN_hidden_2, self.args.DQN_hidden_3,self.args.DQN_output_size).to(self.args.device)
        self.target_model = DQN(self.args.DQN_input_size, self.args.DQN_hidden_1, self.args.DQN_hidden_2, self.args.DQN_hidden_3, self.args.DQN_output_size).to(self.args.device)
        self.DQNMemory = DQNMemory(args)
        #self.critic = Critic(self.args.PPO_input_size, self.args.PPO_hidden_1, self.args.PPO_hidden_2, self.args.PPO_hidden_3).to(self.args.device)
        #self.old_critic = Critic(self.args.PPO_input_size, self.args.PPO_hidden_1, self.args.PPO_hidden_2, self.args.PPO_hidden_3).to(self.args.device)  # Old value network
        #self.old_critic.load_state_dict(self.critic.state_dict())
        #self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.PPO_critic_learningrate)
        self.loss_func = nn.MSELoss()
        self.PPOMemory = PPOMemory(args)
        self.actor_loss = []
        # self.critic_loss = []

    def choose_action(self, s_t, epsi):
        #  Return the action, and the probability to choose this action
        s_t = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(self.args.device)
        with torch.no_grad():
            if self.args.is_test:
                action_prob = self.actor(s_t)
            else:
                action_prob = self.old_actor(s_t)
        c = Categorical(action_prob)
        if self.args.is_test:
            action = action_prob.argmax()
            # action = c.sample()
        else:
            # if random.random() > epsi:
            #     action = action_prob.argmax()
            # else:
            #     #action = c.sample()
            #     action = torch.tensor(random.choice(range(2)))
            action = c.sample()
        # action = c.sample()
        a_log_prob = action_prob[:, action.item()]
        return action.item(), a_log_prob.item()

    def update(self, critic, old_critic):
        s,  s_, a, r, a_old_prob = self.PPOMemory.sample()
        for _ in range(self.K_epoch):
            with torch.no_grad():
                td_target = r + self.discount * old_critic(s)
                td_error = r + self.discount * critic(s_) - critic(s)
                td_error = td_error.detach().cpu().numpy()
                advantage = []  # Advantage Function
                adv = 0.0
                for td in td_error[::-1]:
                    adv = adv * self.LAMBDA * self.discount + td[0]
                    advantage.append(adv)
                advantage.reverse()
                advantage = torch.tensor(advantage, dtype=torch.float).reshape(-1, 1).to(self.args.device)
                # Trick: Normalization
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-7)

            a_new_prob = self.actor(s).gather(1, a)
            ratio = a_new_prob / a_old_prob.detach()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

            actor_loss = (- torch.min(surr1, surr2).mean())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # self.loss.append(actor_loss.item())
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            critic_loss = self.loss_func(td_target.detach(), self.critic(s))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())
        # return self.loss[-1]

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_models(self, i, currentPath):
        # current_dir = dirname(dirname(dirname(os.path.realpath(__file__))))
        model_path = currentPath + self.args.PPO_model_path +str(i-self.args.num_dqn)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.actor.state_dict(), model_path+'_a.ckpt')
        torch.save(self.model.state_dict(), model_path+'_c.ckpt')

    def load_models(self, i, currentPath):
        # current_dir = dirname(dirname(dirname(os.path.realpath(__file__))))
        model_path1 = currentPath + self.args.PPO_model_path +str(i-self.args.num_dqn)
        model_path2 = currentPath + self.args.PPO_model_path +str(i-self.args.num_dqn)
        self.actor.load_state_dict(torch.load(model_path1 + '_a.ckpt'))
        self.model.load_state_dict(torch.load(model_path2 + '_c.ckpt'))