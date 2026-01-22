import torch.nn as nn
import torch.nn.functional as F
from .replay_memory import DQNMemory
import os
from os.path import dirname
import numpy as np
import random
from config import config
import torch
args = config(False)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
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
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        # x = F.relu(self.fc_4(x))
        x = self.fc_4(x)
        return x

# class Rnn(nn.Module):
#     def __init__(self, input_size, n_hidden1, n_hidden2, n_hidden3, output_size):
#         super(Rnn, self).__init__()
#         self.fc_1 = nn.Linear(input_size, n_hidden1)
#         self.fc_1.weight.data.normal_(0, 0.1)
#         self.fc_2 = nn.Linear(n_hidden1, n_hidden2)
#         self.fc_2.weight.data.normal_(0, 0.1)
#         self.rnn = nn.GRUCell(n_hidden2, n_hidden2)
#         self.fc_3 = nn.Linear(n_hidden2, n_hidden3)
#         self.fc_3.weight.data.normal_(0, 0.1)
#         self.fc_4 = nn.Linear(n_hidden3, output_size)
#         self.fc_4.weight.data.normal_(0, 0.1)
#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc_2.weight.new(1, self.args.rnn_hidden_dim).zero_()
#     def forward(self, x,hidden_state):
#         x = F.relu(self.fc_1(x))
#         x = F.relu(self.fc_2(x))
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#         x = F.relu(self.fc_3(h))
#         x = F.relu(self.fc_4(x))
#         return x, h

class DQN_Agent:
    def __init__(self, args):
        self.args = args
        self.discount = self.args.discount
        self.agent_type = 'DQN'
        self.double_q = True
        self.DQNMemory = DQNMemory(args)
        self.model = DQN(self.args.DQN_input_size, self.args.DQN_hidden_1, self.args.DQN_hidden_2,
                         self.args.DQN_hidden_3, self.args.DQN_output_size).to(self.args.device)
        # if torch.cuda.device_count()>1:
        #    self.model = nn.DataParallel(self.model)
        # self.model.to(device)
        self.target_model = DQN(self.args.DQN_input_size, self.args.DQN_hidden_1, self.args.DQN_hidden_2,
                                self.args.DQN_hidden_3, self.args.DQN_output_size).to(self.args.device)  # Target Model
        self.target_model.eval()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.DQN_learning_rate,
                                             momentum=0.05, eps=0.01)
        self.loss_func = nn.MSELoss()
        self.loss = []
        self.hidden_states = None
        #self.update_target_network()
    def choose_action(self, s_t, epsi):
        if self.args.is_test:
            if random.random() > self.args.test_epsilon:
                with torch.no_grad():
                    q_values = self.model(torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(self.args.device))
                    return q_values.max(1)[1].item()
            else:
                self.args.test_epsilon *= self.args.epsilon_decay
                self.args.test_epsilon = max(self.args.epsilon_min, self.args.test_epsilon)
                return random.choice(range(self.args.DQN_output_size))
        else:
            if self.args.epsilon_type == 'exp':
                if random.random() > self.args.train_epsilon:
                    with torch.no_grad():
                        q_values = self.model(torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(self.args.device))
                        return q_values.max(1)[1].item()
                else:
                    self.args.train_epsilon *= self.args.epsilon_decay
                    self.args.train_epsilon = max(self.args.epsilon_min, self.args.train_epsilon)
                    return random.choice(range(self.args.DQN_output_size))
            else:
                if random.random() > epsi:
                    with torch.no_grad():
                        q_values = self.model(torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(self.args.device))
                        return q_values.max(1)[1].item()
                else:
                    return random.choice(range(self.args.DQN_output_size))


    # def choose_action(self, s_t, epsi=0.):
    #     if self.args.is_test:
    #         if random.random() > self.args.test_epsilon:
    #             with torch.no_grad():
    #                 q_values, self.hidden_states=self.model(torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(self.args.device),self.hidden_states)
    #                 return q_values.max(1)[1].item()
    #         else:
    #             self.args.test_epsilon *= self.args.epsilon_decay
    #             self.args.test_epsilon = max(self.args.epsilon_min, self.args.test_epsilon)
    #             return random.choice(range(self.args.DQN_output_size))
    #     else:
    #         if random.random() > epsi:
    #             with torch.no_grad():
    #                 q_values,self.hidden_states=self.model(torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(self.args.device),self.hidden_states)
    #                 return q_values.max(1)[1].item()
    #         else:
    #             # self.args.train_epsilon *= self.args.epsilon_decay
    #             # self.args.train_epsilon = max(self.args.epsilon_min, self.args.train_epsilon)
    #             return random.choice(range(self.args.DQN_output_size))

    # def update(self):  # Double Q-Learning
    #     batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = self.memory.sample()
    #     action = torch.LongTensor(batch_action).to(self.args.device)
    #     reward = torch.FloatTensor(batch_reward).to(self.args.device)
    #     state = torch.FloatTensor(np.float32(batch_s_t)).to(self.args.device)
    #     next_state = torch.FloatTensor(np.float32(batch_s_t_plus_1)).to(self.args.device)
    #     if self.double_q:
    #         next_action = self.model(next_state).max(1)[1]
    #         next_q_values = self.target_model(next_state)
    #         next_q_value = next_q_values.gather(1, next_action.unsqueeze(1)).squeeze(1)
    #         expected_q_value = reward + self.discount * next_q_value
    #     else:
    #         next_q_value = self.target_model(next_state).max(1)[0]
    #         expected_q_value = reward + self.discount * next_q_value
    #     q_values = self.model(state)
    #     q_acted = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    #     loss = self.loss_func(expected_q_value.detach(), q_acted)
    #     # backward and optimize
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.loss.append(loss.item())
    #     return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def init_hidden(self, batch_size):
        self.hidden_states = self.model.init_hidden()

    def save_models(self, i, currentPath):
        # current_dir = dirname(dirname(dirname(os.path.realpath(__file__))))
        model_path = currentPath + self.args.DNQ_model_path + str(i)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.model.state_dict(), model_path+'.ckpt')
        torch.save(self.target_model.state_dict(), model_path+'_t.ckpt')
        print('Agent' + str(i) + 'saved!')

    def load_models(self, i, currentPath):
        # current_dir = dirname(dirname(dirname(os.path.realpath(__file__))))
        model_path = currentPath + self.args.DNQ_model_path + str(i)
        self.model.load_state_dict(torch.load(model_path + '.ckpt'))
        self.target_model.load_state_dict(torch.load(model_path + '_t.ckpt'))