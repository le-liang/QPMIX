import numpy as np
import envsettings as env
# from qagent import QAgent
from PacketManager import PacketManager
import Arg
from copy import deepcopy
import random
from config import config
import torch
args = config(False)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
import pdb

class AgentOCA:
    def __init__(self,
                args,
                id=0,
                batch_size=32,
                learning_rate=0.001,
                epochs=1,
                mac=None,
                fairnessWeight=0.8,
                traffic_type='poisson',
                packet_length=1080e-6,
                epsilon=Arg.epsilon,
                epsilon_min=Arg.epsilon_min,
                epsolin_decay=Arg.epsilon_decay):
        self.id =id
        # parameters
        self.args = args
        self.SlotTime = int(9e-6 / env.timestep)
        self.packetLength = int(packet_length / env.timestep)
        self.ExtraWait = 0
        self.memoryThreshold = 200
        self.actionProsponed = 0
        self.numactionProsponed = self.args.numactionProsponed[self.args.n_agents] if args.is_test else 0
        self.fairnessWeight = fairnessWeight
        self.maxQueuelen = 10
        self.AvgTraffic = env.lmda
        self.state_size = self.args.observation_size

        # mac initilization
        self.mac = mac
        self.agent_eval=0

        # initialization
        # action, state, reward
        self.action = 0
        self.agent_state = np.zeros(self.state_size)
        # The self.agent_next_state is actually the current state
        self.agent_next_state = np.zeros(self.state_size)
        # self.agent_observation is the observation of the channel state
        self.agent_observation = 0
        self.agent_reward = []
        self.is_transmitting = False
        self.oldProb = None
        # packet processing
        self.packetManager = PacketManager(self.maxQueuelen, traffic_type=traffic_type)
        self.packetGenCount = 0
        self.totalpaketSuccess = 0
        self.packetSuccess = 0
        # Counters
        self.action_flag = -1
        self.iterCount = 0
        # The time it takes for a node's state machine to move from the current state to the next state
        self.counter = 0
        self.packetSuccess = 0
        # Counters
        self.action_flag = -1
        self.iterCount = 0
        self.counter = 0
        self.ChIdleCount = 0
        # D2LT of self and other nodes
        self.slot2lastTransmit = np.zeros(2)
        # Tracking lists
        self.rewardTrack = []
        self.actionTrack=[]
        self.successTrack=[]
        self.generateTrack=[]
        self.evalTrack = []
        # Dicts
        self.takeAction={
            0: self.wait,
            1:self.transmit,
        }

        self.ret = {0: 'WAIT', 1:'Transmit'}
        self.stateIdleToBusy = None
        self.oldProbIdleToBusy = None
    def step(self, Channel, epsi=0.):
        self.slot2lastTransmit += 1
        # Whether new packet comes
        newPacket = self.packetManager.newPacket(time=self.iterCount, avgtraffic=self.AvgTraffic)
        self.packetGenCount +=newPacket
        # Actions according to current state
        self.takeAction[self.action](Channel, epsi)
        self.iterCount +=1
        return self.ret[self.action]

    def wait(self, Channel, epsi):
        if self.counter == 0:
            if Channel.state == "IDLE":
                if self.totalpaketSuccess < Channel.packetSuccess:
                    # pdb.set_trace()
                    # update the number of packets successfully transmitted by the channel
                    self.totalpaketSuccess = Channel.packetSuccess
                    # Set the D2LT of other nodes to 0
                    self.slot2lastTransmit[1] = 0
                    self.to_learn = True
                if self.is_transmitting:
                    self.slot2lastTransmit[0] = 0
                    self.is_transmitting = False
                self.agent_observation = 0
            else:
                # Wait and someones are transmitting
                self.agent_observation = 1 # Someone use the Channel
            # self.updateState(self.SlotTime)
            # Determine action
            if Channel.state == 'IDLE' and not self.packetManager.empty():
                if (self.ChIdleCount >=self.ExtraWait):
                    if self.args.isSharePara:
                        self.action = self.mac.choose_action(self.agent_state, self.id)
                    else:
                        if self.args.agents[self.id] == 'DQN':
                            if self.actionProsponed < self.numactionProsponed:
                                if self.actionProsponed % self.args.n_agents == self.id:
                                    self.action = 1
                                else:
                                    self.action = 0
                                self.actionProsponed += 1
                            else:
                                self.action = self.mac.choose_action(self.agent_state, self.id, epsi)
                                self.stateIdleToBusy = deepcopy(self.agent_state)
                        else:
                            if self.actionProsponed < self.numactionProsponed:
                                if self.actionProsponed % self.args.n_agents == self.id:
                                    self.action = 1
                                    #self.action, self.oldProb = self.mac.choose_action(self.agent_state, self.id, epsi)
                                else:
                                    self.action = 0
                                    self.oldProb = 1
                                self.actionProsponed += 1
                            else:
                                self.action, self.oldProb = self.mac.choose_action(self.agent_state, self.id, epsi)
                                self.stateIdleToBusy = deepcopy(self.agent_state)
                                self.oldProbIdleToBusy = deepcopy(self.oldProb)

                else:
                    self.action = 0
                    self.oldProb = 1.0
                    self.action_flag = -1
                if self.action == 1:
                    self.counter = self.packetLength -1
                    self.ChIdleCount = 0
                else:
                    self.counter = self.SlotTime - 1
                    self.ChIdleCount += 1
            else:
                self.oldProb = 1.0
                self.counter = self.SlotTime -1
                self.ChIdleCount = 0
        else:
            self.counter -= 1

    def transmit(self, Channel, epsi=0., actionProposed=0):
        if self.counter == 0:
            # Decide observation and reward at the end of the transmission
            if not Channel.collision:
                # Successfully transmitted
                self.agent_observation = 0
                self.packetManager.success(time=self.iterCount)
                self.packetSuccess += 1
                self.totalpaketSuccess += 1
                self.slot2lastTransmit[0] = 0
                self.is_transmitting = True
            else:
                # Transmit failed
                self.packetManager.fail(time=self.iterCount)
                self.agent_observation = 1

            # Neither the action nor the status changed during transmission, only the duration changed
            # self.updateState(self.packetLength)
            self.to_learn = True
            self.action = 0
            self.counter = self.SlotTime - 1
        else:
            self.counter -= 1

    def updateState(self, duration, STAstates, id):
        time_delay = self.slot2lastTransmit.copy()
        if STAstates.count('Transmit') == 1:
            if STAstates[id] == 'Transmit':
                time_delay[1] += duration
                time_delay[0] = 0
            else:
                time_delay[1] = 0
                time_delay[0] += duration
        elif STAstates.count('Transmit') > 1:
            time_delay += duration
        time_delay = time_delay / (np.sum(time_delay) + 1e-10)
        observation = 1 if sum(1 for idx, state in enumerate(STAstates) if idx != id and state == 'Transmit') > 0 else 0
        self.agent_state = np.concatenate((self.agent_state[5:], [self.action, observation,
                                                                                 duration/ self.packetLength], time_delay))
    def getState(self):
        return self.agent_state

    def reset(self):
        self.action = 0
        self.agent_state = np.zeros(self.state_size, int)
        self.agent_observation = 0
        self.slot2lastTransmit = np.zeros(2)
        self.counter = 0
    def save(self, filepath=''):
        np.save(filepath + '/agent_' + str(self.id) + '_actionTrack.npy', np.array(self.actionTrack))
        np.save(filepath + '/agent_' + str(self.id) + '_successTrack.npy', np.array(self.successTrack))
        np.save(filepath + '/agent_' + str(self.id) + '_successTrack.npy', np.array(self.successTrack))
        np.save(filepath + '/agent_' + str(self.id) + '_evalTrack.npy', np.array(self.evalTrack))
        self.packetManager.save(filepath + '/agent_' + str(self.id))
