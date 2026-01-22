import numpy as np
import envsettings as env
import random
from config import config
import torch
args = config(False)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
class PacketManager:
    def __init__(self, maxLength= 10, traffic_type='poisson'):
        self.maxLength = maxLength
        self.packetQueue = Queue(maxsize=maxLength)
        self.dropList = []
        self.successList = []
        self.failList = []
        self.packetID = 0
        self.VoIP_counter = 0
        self.traffic_type = traffic_type
        self.rng = np.random.default_rng(args.seed)
        self.totalPacket = 0

    def newPacket(self, time=0, avgtraffic = env.lmda):
        if self.traffic_type == "VoIP":
            VoIP_counter_next = self.VoIP_counter + 50 * env.timestep
            num = int(np.floor(VoIP_counter_next)-np.floor(self.VoIP_counter))
            self.VoIP_counter = VoIP_counter_next
        else:
            num = int(self.rng.poisson(avgtraffic,1))
            self.totalPacket += num
        for i in range(num):
            if self.packetQueue.full():
                # self.dropList.append([self.packetID, time])
                self.packetID += 1
            else:
                self.packetQueue.put([self.packetID, time])
                self.packetID += 1
        return num

    def popPacket(self,success, time=0, num=1):
        for i in range(num):
            if not self.packetQueue.empty():
                Packet = self.packetQueue.get()
                Packet.append(time)
                if success:
                    self.successList.append(Packet)
                else:
                    self.failList.append(Packet)

    def success(self, time=0, num=1):
        self.popPacket(success=True, time=time, num=num)

    def fail(self, time=0, num=1):
        self.popPacket(success=False, time=time, num=num)

    def full(self):
        return self.packetQueue.full()

    def empty(self):
        return self.packetQueue.empty()

    def save(self, filepath):
        np.save(filepath+'_successList.npy', np.array(self.successList))
        np.save(filepath + '_dropList.npy', np.array(self.dropList))
        np.save(filepath + '_failList.npy', np.array(self.failList))

class Queue:
    def __init__(self, maxsize = 10):
        self.content = []
        self.maxsize = maxsize

    def empty(self):
        if len(self.content) == 0:
            return True
        else:
            return False

    def full(self):
        if len(self.content) == 10:
            return True
        else:
            return False

    def put(self, item):
        if self.full():
            raise RuntimeError('full')
        else:
            self.content.append(item)
    def get(self):
        if self.empty():
            raise RuntimeError('empty')
        else:
            return self.content.pop(0)