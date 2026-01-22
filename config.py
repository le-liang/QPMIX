import torch
from pettingzoo.mpe import simple_spread_v3
import os
import math
import envsettings as env
import os.path
from torch.optim import RMSprop
import numpy as np

# isAgentSharePara = False
# isDDQN = True
# isDueling = False
# GRU_num = 128
# qmix_hidden_dim = 128
# hyper_hidden_dim = 128
#
# numOCA = 10
# state_size = 5*15
# batch_size = 64
# gamma = 0.5
# update_interval = 100 #target network update freq
# memory_size = 2000
# memoryThreshold = 500

class config:
    def __init__(self, mode):
        #environment config
        #self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.dir = './model'
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        self.train_num_episode = 2000
        self.train_num_step = 100
        self.test_num_episode = 1
        self.test_num_step = 10000
        self.agent_nor = 'sum'
        self.mixing_nor = 'sum'
        self.rewardOption = 'onePlus'
        self.is_test = mode
        self.agents = 2 * ['DQN'] + 2 * ['PPO']
        self.num_dqn = self.agents.count('DQN')
        self.num_ppo = self.agents.count('PPO')
        self.isSharePara = False
        self.epsilon_type = 'linear' # exp or linear

        self.n_agents = len(self.agents)
        self.weights = 1
        self.n_actions = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.M = 5 # state length
        self.observations_length = 10
        self.mac = 'basic_mac'
        self.observation_size = self.observations_length * self.M
        # self.input_shape = self.M * self.observations_length + self.n_agents
        self.seed = 1
        # QMIX config
        self.mixing_embed_dim = 16
        self.hypernet_embed = 64
        self.state_shape = 2
        self.discount = 0.5
        self.qmix_lr = 5e-4
        self.optim_alpha = 0.99
        self.optim_eps = 0.00001
        self.qmix_max_grad_norm = 0.5
        self.s_t_input_size = self.n_agents*2
        self.beta = 1
        self.hypernet_layers = 2
        self.numactionProsponed = [0, 0, 0, 1, 4, 5, 5, 1, 1, 5]

        # DQN config
        self.DQN_input_size = self.observation_size
        self.DQN_hidden_1 = 250
        self.DQN_hidden_2 = 120
        self.DQN_hidden_3 = 120
        self.DQN_output_size = self.n_actions
        self.DQN_memory_size = 500
        self.DQN_batch_size = 32
        self.DNQ_model_path = '/heterogeneous'+str(self.n_agents)+'/mixDQN' + str(self.num_dqn) + '/agent_'
        self.IDQN_model_path = '/mix' + str(self.n_agents) + '/DQN' + str(self.num_dqn) + '/agent_'
        self.DQN_fig_path = './fig/' + 'heterogeneous'+str(self.n_agents)+'/mixDQN' + str(self.num_dqn)
        self.DQN_data_path = './data/' + 'heterogeneous'+str(self.n_agents)+'/mixDQN' + str(self.num_dqn)
        self.DQN_learning_rate = 1e-5
        self.train_epsilon = 1
        self.test_epsilon = 0.0
        self.epsi_anneal_length = int(math.ceil(env.simulationTime / env.timestep))
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.998
        self.mini_batch_step = 50
        self.target_update_step = self.mini_batch_step * 5
        self.double_q = True

        # PPO config
        self.PPO_input_size = self.observation_size
        self.PPO_hidden_1 = 250
        self.PPO_hidden_2 = 120
        self.PPO_hidden_3 = 120
        self.PPO_output_size = self.n_actions
        self.PPO_model_path = '/heterogeneous' +str(self.n_agents) +'/mixPPO' + str(self.num_ppo) + '/agent_'
        self.IPPO_model_path = '/mix' +str(self.n_agents) + '/PPO' + str(self.num_ppo) + '/agent_'
        self.PPO_fig_path = './fig/' + 'heterogeneous' +str(self.n_agents) +'/mixPPO' + str(self.num_ppo)
        self.PPO_data_path = './data/' + 'heterogeneous' +str(self.n_agents) +'/mixPPO' + str(self.num_ppo)
        self.PPO_actor_lr = 1e-5
        self.PPO_V_lr = 1e-4
        self.K_epoch = 8
        self.LAMBDA = 0.95
        self.gamma = 1
        self.clip_param = 0.2
        self.PPO_max_grad_norm = 0.5
        self.entropy = 0.2


