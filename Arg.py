import os.path
from torch.optim import RMSprop
import numpy as np
dir = './model'

if not os.path.exists(dir):
    os.mkdir(dir)

isAgentSharePara = False
isDDQN = True
isDueling = False
GRU_num = 128
qmix_hidden_dim = 128
hyper_hidden_dim = 128

numOCA = 10
state_size = 5*15
batch_size = 64
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.998
gamma = 0.5
update_interval = 100 #target network update freq
memory_size = 2000
memoryThreshold = 500

agent_nor = 'sum'
mixing_nor = 'sum'
rewardOption = 'onePlus'
weights = np.sqrt(numOCA)