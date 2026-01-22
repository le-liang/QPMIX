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
from tqdm import tqdm
from Plot import Plot
import random
import torch
args = config(False)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
def run_MultiAgent(args, numOCA=1, numCSMA=1, w=1, traffic_type = [], packet_length=[], reward_Option = 'onePlus'):
    currentPath = args.dir + '/numOCA={}'.format(numOCA)
    CSMA_CA_path = args.dir + '/CSMA_CA={}'.format(numCSMA)
    if not os.path.exists(currentPath):
        os.mkdir(currentPath)
    plot = Plot(currentPath, args)
    STA = []
    if traffic_type == []:
        traffic_type = ['poisson'] * (numOCA + numCSMA)
    if packet_length == []:
        packet_length = [1080e-6] * (numOCA + numCSMA)
    numSTA = numOCA + numCSMA
    mac = controller[args.mac](args)
    mac.load_models(currentPath)
    if args.isSharePara:
        state_size = args.observations_size
        qagent = QAgent(
            n_actions = 2,
            n_agents = numOCA,
            state_size = state_size,
            epsilon = args.train_epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay
        )
        for i in range(numOCA):
            STA.append(
                AgentOCA(
                    args,
                    id = i,
                    mac =mac,
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

    time0 = time.time()

    currentTime = 0
    progress = 0
    round_time = []
    rewardTrack = []
    flag = 0 # for debug
    itration = math.ceil(env.testTime/ env.timestep)
    Ch = Channel(numSTA, itration, args, STA, mac, reward_Option)
    for _ in tqdm(range(itration)):
        time2 = time.time()
        STAstates = []
        STAactions = []
        STAobs = []
        oldProbs = []
        for i in range(numSTA):
            # Observation
            STAobs.append(STA[i].getState())
            Ch.preSlot2lastTransmit[i] = STA[i].slot2lastTransmit[0]
            STAstates.append(STA[i].step(Ch))
            Ch.slot2lastTransmit[i] = STA[i].slot2lastTransmit[0]
            # Action
            STAactions.append(0 if STAstates[i] == 'WAIT' else 1)
            # Probility of PPO
            oldProbs.append(STA[i].oldProb)

        if Ch.state == 'IDLE':
            # Global state
            state = Ch.getGlobalState()
            rewards = Ch.step(STAactions)
            rewardTrack.append(rewards)
            if STAstates.count('Transmit') == 0:
                duration = 1
            else:
                duration = packet_length[0] / env.timestep
            for i in range(numSTA):
                STA[i].updateState(duration=duration, STAstates=STAstates, id=i)
        Ch.updateState(STAstates)

        # currentTime += env.timestep
        if (currentTime/env.simulationTime) >= progress / 100:
            progress += 10
        time3=time.time()
        round_time.append(time3-time2)

    time1 = time.time()
    print(numSTA, 'STAs transmitted:', Ch.packetSuccess, 'packets')
    print('Total throughput:', Ch.packetSuccess * packet_length[0] / env.testTime)
    for i in range(numSTA):
        print('STA', i, 'transmitted:', STA[i].packetSuccess)
        print('STA', i, 'throughput:', STA[i].packetSuccess * packet_length[i] / env.testTime)
        print('STA', i, 'Total Packet:', STA[i].packetManager.totalPacket)
        print('STA', i, 'collision rate:', Ch.collisionCount[i] / Ch.packetCount[i])
        print('\n')
    print('Total Runtime:', (time1 - time0))
    # for i in range(numOCA):
    #     STA[i].save(currentPath)
    # for i in range(numCSMA):
    #     STA[i+numOCA].save(CSMA_CA_path)
    # np.save(currentPath+'/rount_time.npy',np.array(round_time))
    # Ch.save(currentPath)
    # plot.plotThoughput()
    # meanDelay, delayJitter, mean_delays_per_STA = plot.OCA_delay(currentPath, STA)
    # print('meanDelay:', meanDelay)
    # print('delayJitter:', delayJitter)
    # print('mean_delays_per_STA:', mean_delays_per_STA)
    return currentPath, (time1-time0)/60.0

if __name__ == '__main__':
    # for numOCA in range(16,17):
    args = config(mode=True)
    numCSMA = 0
    numOCA = args.n_agents
    # numOCA = 0
    traffic_type = ['poisson'] * (numOCA+numCSMA)
    # traffic_type = ['VoIP'] * (numOCA+numCSMA)
    run_MultiAgent(args, numOCA=numOCA, numCSMA=numCSMA, w=1,
                   traffic_type=traffic_type, packet_length=[1080e-6]*(numOCA+numCSMA))