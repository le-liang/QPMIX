import os
import math
import time
import numpy as np
import envsettings as env
from AgentOCA import AgentOCA
from Channel import Channel
from CSMA_CA_noACK import CSMA_CA
from qagent_share import QAgent
from controllers import REGISTRY as controller
from config import config
import random
import torch
from tqdm import tqdm
from Plot import Plot


args = config(False)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
def run_MultiAgent(args, numOCA=1, numCSMA=1, w=1, traffic_type = [], packet_length=[], reward_Option = 'onePlus'):
    currentPath = args.dir + '/numOCA={}'.format(numOCA)
    if not os.path.exists(currentPath):
        os.mkdir(currentPath)
    STA = []
    plot = Plot(currentPath, args)

    if traffic_type == []:
        traffic_type = ['poisson'] * (numOCA + numCSMA)
    if packet_length == []:
        packet_length = [1080e-6] * (numOCA + numCSMA)
    numSTA = numOCA + numCSMA
    mac = controller[args.mac](args)
    if args.isSharePara:
        for i in range(numOCA):
            STA.append(
                AgentOCA(
                    args,
                    id = i,
                    mac = mac,
                    batch_size = 32,
                    fairnessWeight = w,
                    traffic_type = traffic_type[i],
                    packet_length = packet_length[i]
                )
            )
    else:
        for i in range(numOCA):
            STA.append(
                AgentOCA(
                    args,
                    id=i,
                    batch_size=32,
                    mac=mac,
                    fairnessWeight=w,
                    traffic_type=traffic_type[i],
                    packet_length=packet_length[i]
                )
            )
    for i in range(numOCA,numSTA):
        STA.append(CSMA_CA(id=i,traffic_type=traffic_type[i],packet_length=packet_length[i]))
    episode = math.ceil(env.simulationTime)
    itration = math.ceil(env.simulationTime/ env.timestep)
    Ch = Channel(numOCA, itration, args, STA, mac, reward_Option)

    time0 = time.time()

    currentTime = 0
    progress = 0
    round_time = []
    idleCount = 0
    rewardTrack = []
    state = np.array([0] * args.n_agents * 2, dtype=np.float32)
    flag = 0 # for debug
    # for j in tqdm(range(episode)):
    #     for agent in STA:
    #         agent.reset()
    #         idleCount = 0
    #         state = np.array([0] * args.n_agents * 2, dtype=np.float32)
    for it in tqdm(range(itration)):
        time2 = time.time()
        STAstates = []
        STAactions = []
        STAobs = []
        STANextobs = []
        oldProbs = []
        SLOT2LAST = []
        if args.epsilon_type == 'linear':
            if it < args.epsi_anneal_length:
                epsi = 1 - (it) * (1 - args.epsilon_min) / (args.epsi_anneal_length - 1)  # epsilon decreases over each episode
            else:
                epsi = args.epsilon_min
        else:
            epsi = 0.01

        for i in range(numSTA):
            # Observation
            STAobs.append(STA[i].getState())
            Ch.preSlot2lastTransmit[i] = STA[i].slot2lastTransmit[0]
            STAstates.append(STA[i].step(Ch, epsi))
            Ch.slot2lastTransmit[i] = STA[i].slot2lastTransmit[0]
            # Action
            STAactions.append(0 if STAstates[i] == 'WAIT' else 1)
            # Probility of PPO
            oldProbs.append(STA[i].oldProb)
        if Ch.state == 'IDLE':
            idleCount += 1
            # Global state
            # state = Ch.getGlobalState()
            rewards = Ch.step(STAactions)
            # rewardTrack.append(rewards)
            if STAstates.count('Transmit') == 0:
                duration = 1
            else:
                duration = packet_length[0] / env.timestep
            for i in range(numSTA):
                STA[i].updateState(duration=duration, STAstates=STAstates, id=i)
                # Next observation
                STANextobs.append(STA[i].getState())
                SLOT2LAST.append(STANextobs[i][-2])
            # Next global state
            # nextState = np.concatenate((STAactions, [STANextob[-2] for STANextob in STANextobs]))
            # nextState = np.concatenate((STAactions, STANextobs[0][-2:]))
            nextState = np.concatenate((STAactions, SLOT2LAST/np.sum(SLOT2LAST) + 1e-10))
            # nextState = np.concatenate((STAactions, SLOT2LAST))
            mac.insert(STAobs, STANextobs, STAactions, rewards, state, nextState, oldProbs)
            state = nextState
            if 'DQN' in args.agents and (mac.agents[0].DQNMemory.count > args.DQN_batch_size):
                if idleCount % 10 == 0:
                    mac.DQN_train()
                if idleCount % 1000 == 0:
                    mac.update_target_network()
            if 'PPO' in args.agents and (mac.agents[-1].PPOMemory.count > 30):
                mac.PPO_train()
        Ch.updateState(STAstates)
    time1 = time.time()
    print(numSTA, 'STAs transmitted', Ch.packetSuccess, 'packets')
    print('Total throughput:', Ch.packetSuccess * packet_length[0]/env.simulationTime)
    for i in range(numSTA):
        print('STA',i, 'transmitted', STA[i].packetSuccess)
        print('STA',i, 'throughput', STA[i].packetSuccess * packet_length[i]/env.simulationTime)
    print('Total Runtime',(time1-time0))

    # for i in range(numSTA):
    #     STA[i].save(currentPath)
    mac.save_models(currentPath)
    # Ch.save(currentPath)
    # np.save(currentPath+'/rount_time.npy',np.array(round_time))
    STA = []
    return currentPath, (time1-time0)/60.0

if __name__ == '__main__':
    # for numOCA in range(16,17):
    args = config(mode=False)
    numCSMA = 0
    numOCA = args.n_agents
    traffic_type = ['poisson'] * (numOCA+numCSMA)
    run_MultiAgent(args, numOCA=numOCA, numCSMA=numCSMA, w=1,
                   traffic_type=traffic_type, packet_length=[1080e-6]*(numOCA+numCSMA))