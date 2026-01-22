import numpy as np
import time
import pdb
import random
from config import config
import torch
args = config(False)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
class Channel:
    def __init__(self,numOCA, iteration, args, STA, mac, rewardOption='onePlus'):
        self.state = 'IDLE'
        self.collision = False
        self.packetSuccess = 0
        self.iterCount = 0
        self.stateTrack = []
        self.args = args
        self.collisionTrack = []
        self.iteration = iteration
        self.successTrack = []
        self.mac = mac
        self.collisionCount = np.zeros(numOCA, int)
        self.suceessCount = np.zeros([numOCA, self.iteration+1], int)
        self.packetCount = np.zeros(numOCA, int)
        self.slot2lastTransmit = np.zeros(numOCA)
        # self.qimx = QMix(agents=STA, batch_size=Arg.batch_size, gamma=Arg.gamma)
        self.agents = STA
        self.n_agents = args.n_agents
        self.reward = None
        # self.obs_size = self.agents[0].state_size
        self.lastRewardCount = 0
        self.rewardCount = 0

        self.agent_action_num = args.n_actions
        self.lossTrack = []
        # self.rewardTrack = []
        self.rewardTrack = np.zeros([self.iteration,1])
        self.trainCount = 0
        self.rewardOption = rewardOption
        self.preSlot2lastTransmit = np.zeros(numOCA)
        self.updateCount = 0
        self.act = np.zeros(self.n_agents, int)
    def step(self, STAactions):
        self.act = STAactions
        reward = self.rewardFunc()
        # self.rewardCount = self.iterCount
        # self.rewardTrack[self.lastRewardCount:self.rewardCount] = self.reward[-1]
        # self.lastRewardCount = self.rewardCount
        return reward
    def updateState(self, STAstates):
        """
        Update channel state, collision, packet success, and other info
        The channel state is updated after all STAs choose their actions
        :param STAstates: list of STA states:Wait or Transmit
        :return: None
        """
        self.iterCount += 1
        if self.state == 'IDLE':
            if STAstates.count('Transmit') !=0:
                self.state = 'BUSY'
                self.trans_index = np.where(np.array(STAstates) == 'Transmit')[0]
        if self.state == 'BUSY':
            # wait_index = np.where(STAstates == 'Wait')[0]
            if STAstates.count('Transmit') == 0:
                # Transmit finished
                if not self.collision:
                    self.packetSuccess += 1
                    self.suceessCount[self.trans_index, self.iterCount] = 1
                else:
                    self.collisionCount[self.trans_index] += 1
                self.state = 'IDLE'
                self.collision = False
                self.packetCount[self.trans_index] += 1
            if STAstates.count('Transmit') > 1:
                self.collision = True

        # self.stateTrack.append((self.iterCount, self.state))
        # self.collisionTrack.append((self.iterCount, self.collision))
        # self.successTrack.append((self.iterCount, self.packetSuccess))

    def oneHotAciton(self, actions, n_agents, n_actions):
        one_hot_vector = np.zeros([n_agents, n_actions])
        wait_index = np.where(actions == 0)[0]
        trans_index = np.where(actions == 1)[0]
        one_hot_vector[wait_index, 0] = 1
        one_hot_vector[trans_index, 1] = 1
        return one_hot_vector

    def getGlobalState(self):
        return np.concatenate((self.act, self.preSlot2lastTransmit/np.sum(self.preSlot2lastTransmit + 1e-10)))
    def learn(self, observations, actions, oldProbs, it):
        # State info
        if self.args.mixing_nor == 'sum':
            time_delay = self.preSlot2lastTransmit/(np.sum(self.preSlot2lastTransmit) + 1e-10)
        elif self.args.mixing_nor == 'linear':
            time_delay = self.preSlot2lastTransmit /1000
        # The global state info
        state_info = np.concatenate((self.act.copy(), time_delay))
        self.act = np.array(actions)
        # actions = self.oneHotAciton(actions, self.n_agents, self.agent_action_num)
        # for i in range(self.n_agents):
        #     action_id = actions[i]
        #     # Transform actions to one-hot
        #     actions[i] = [0] * self.agent_action_num
        #     # Add one-hot agent_id to observation
        #     # if self.args.isSharePara:
        #     #     add_func = self.agents[0].agent.add_agent_id
        #     #     observations[i] = add_func(i, observations[i])
        #     #     observations[i], observations[i].flatten()
        self.reward = self.rewardFunc()
        observations = np.array(observations)
        self.mac.insert(observations, actions, self.reward.copy(), state_info, oldProbs)
        # if self.rewardOption == 'onePlus':
        #     self.qimx.store_transitiom(state_info, observations,actions, self.reward)
        # else:
        #     self.qimx.store_transitiom(state_info, observations, actions, np.array([self.reward]))
        #
        # if self.qimx.memory_counter > self.qimx.memoryThreshold:
        #     loss = self.qimx.train()
        #     self.lossTrack.append((self.iterCount, loss))
        if 'DQN' in self.args.agents and (self.mac.agents[0].DQNMemory.count > self.args.DQN_batch_size):
            if self.updateCount % 1 == 0:
                self.mac.DQN_train()
                # self.mac.PPO_train()
                self.updateCount += 1
            if self.updateCount % 1000 == 0:
                self.mac.update_target_network()
        if 'PPO' in self.args.agents and (self.mac.agents[-1].PPOMemory.count > 20):
            self.mac.PPO_train()
        #self.rewardTrack.append((self.iterCount, self.reward[-1]))
        self.rewardCount = self.iterCount
        self.rewardTrack[self.lastRewardCount:self.rewardCount] = self.reward[-1]
        self.lastRewardCount = self.rewardCount
    def test(self, actions):
        self.act = np.array(actions)
        self.reward = self.rewardFunc()
        # self.rewardTrack.append((self.rewardCount, self.reward[-1]))
        self.rewardCount = self.iterCount
        self.rewardTrack[self.lastRewardCount:self.rewardCount] = self.reward[-1]
        self.lastRewardCount = self.rewardCount

    def rewardFunc(self):
        if self.rewardOption == 'gzy1':
            # Obtain 1 if the agent with max d transmits successfully , otherwise -1
            if np.sum(self.preSlot2lastTransmit[self.preSlot2lastTransmit == np.max(self.preSlot2lastTransmit)]):
                reward = 1
            else:
                reward = -1

        elif self.rewardOption == 'gzy2':
            r_nor = self.preSlot2lastTransmit /1000
            if np.sum(self.act) == 1:
                reward = r_nor[np.argmax(self.act)]
            elif np.sum(self.act) > 1:
                reward = -np.max(r_nor)
            else:
                reward = 0

        elif self.rewardOption == 'gzy3':
            r_nor = self.preSlot2lastTransmit / np.sum(self.preSlot2lastTransmit + 1e-10)
            if np.sum(self.act) == 1:
                if np.argmax(self.act) in np.arange(# The agent with max d success
                    len(self.preSlot2lastTransmit[self.preSlot2lastTransmit == np.max(self.preSlot2lastTransmit)])):
                    reward = 1
                else: # Other agent success
                    reward = -1 #r_nor[np.argmax(self.act)] -1
            elif np.sum(self.act) > 1:
                reward = -1
            else:
                reward = 0

        elif self.rewardOption == 'onePlus':
            r_nor = self.preSlot2lastTransmit / np.sum(self.preSlot2lastTransmit + 1e-10)
            reward = np.zeros(self.n_agents + 1)
            for i in range(self.n_agents):
                if i in np.argwhere(self.preSlot2lastTransmit == np.amax(self.preSlot2lastTransmit)): # Optimal action of agent i is to transmit
                #if i == np.argmax(self.preSlot2lastTransmit):
                    if self.act[i] == 1: # Indeed transmit
                        reward[i] = 1
                    else: # Actually not transmit
                        reward[i] = -1
                else: # Optimal action of agent i is not to transmit
                    if self.act[i] == 0: #indeed not transmit
                        reward[i] = 1
                    else: # Actuall transmit
                        reward[i] = -1

            if np.sum(self.act) == 1:
                sorted_indices = np.argsort(self.preSlot2lastTransmit)[::-1]
                top_3_values = self.preSlot2lastTransmit[sorted_indices[:2]]
                top_3_indices = []
                for value in top_3_values:
                    indices = np.flatnonzero(self.preSlot2lastTransmit == value)
                    top_3_indices.extend(indices)
                # if np.argmax(self.act) in top_3_indices:

                if np.argmax(self.act) in np.argwhere(self.preSlot2lastTransmit == np.amax(self.preSlot2lastTransmit)):
                    reward[-1] = 1
                else:  # other agent success
                    reward[-1] = -1 #(r_nor[np.argmax(self.act)] - 1)
            elif np.sum(self.act) > 1:
                reward[-1] = -1
            else: # No transmit
                reward[-1] = -1
            reward[-1] *= self.args.weights
        else:
            raise RuntimeError('No such reward option!')
        return reward

    def save(self, filepath=''):
        # self.qimx.save(filepath+ '/Channle')
        np.save(filepath + '/ChannelStateTrack.npy', np.array(self.stateTrack))
        np.save(filepath + '/ChannelsuccessTrack.npy', np.array(self.successTrack))
        np.save(filepath + '/ChannelcollisionTrack.npy', np.array(self.collisionTrack))
        np.save(filepath + '/ChannellossTrack.npy', np.array(self.lossTrack))
        np.save(filepath + '/ChannelrewardTrack.npy', np.array(self.rewardTrack))
        np.save(filepath + '/suceessCount.npy', self.suceessCount)
        print('Channel saved!')
