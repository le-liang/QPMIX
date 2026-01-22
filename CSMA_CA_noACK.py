import numpy as np
import envsettings as env
from PacketManager import PacketManager

class CSMA_CA:
    def __init__(self, id =0,traffic_type = 'poisson', packet_length=1080e-6):
        self.id = id

        #para
        self.SlotTime = int(9e-6/env.timestep)
        self.packetLen = int(packet_length/env.timestep)
        self.DIFS = int(36e-6/env.timestep)
        self.SIFS = self.DIFS - 2*self.SlotTime
        self.categories = ['AV_VO', 'AC_VI', 'AC_BE']
        self.categorie = 0 # 0 for AV_VO, 1 for AC_VI, 2 for AC_BE
        self.maxBEB = 1  # 1 for AV_VO, 1 for AC_VI, 5 for AC_BE
        self.initCW = 7    # 7 for AV_VO, 15 for AC_VI, 31 for AC_BE
        self.maxQueueLen = 10
        self.AvgTraffic = env.lmda

        #init
        self.BEB = 0
        self.state = 'IDLE'

        #packet processing
        self.rng = np.random.default_rng()
        self.packetManager = PacketManager(self.maxQueueLen, traffic_type=traffic_type)
        self.packetGenCount = 0
        self.packetSuccess = 0
        #counter
        self.iterCount = 0
        self.counter = 0
        self.counterContention = 0
        #tracking list
        self.generateTrack = []
        self.actionTrack = []
        self.successTrack = []
        #dicts
        self.fsm = {
            'IDLE': self.idle,
            'DIFS': self.difs,
            'Contention': self.contention,
            'Transmit': self.transmit,
            'HOLD': self.hold,
        }
    def step(self, Channel):
        # Whether new packet comes
        newPacket = self.packetManager.newPacket(time=self.iterCount, avgtraffic=self.AvgTraffic)
        self.packetGenCount += newPacket
        # Actions according to current state
        self.fsm[self.state](Channel)
        self.generateTrack.append((self.iterCount,self.packetGenCount))
        self.actionTrack.append((self.iterCount,self.state))
        self.successTrack.append((self.iterCount,self.packetSuccess))
        self.iterCount +=1
        return self.state

    def idle(self, Channel):
        if not self.packetManager.empty():
            self.state = 'DIFS'
            self.counter = self.DIFS -1

    def difs(self, Channel):
        if self.counter ==0:
            self.state = 'Contention'
            self.counter = self.SlotTime -1
            if self.counterContention ==0:
                self.counterContention = np.random.randint(self.initCW * 2**self.BEB)+1
        elif (self.counter % self.SlotTime==0) and (self.counter<=self.DIFS-self.SIFS):
            if Channel.state == 'IDLE':
                self.counter -= 1
            else:
                self.counter = self.DIFS - 1
        else:
            self.counter -= 1

    def contention(self, Channel):
        if self.counter == 0:
            self.counterContention -= 1
            if self.counterContention == 0:
                if Channel.state =='IDLE':
                    self.state = 'Transmit'
                    self.counter=self.packetLen-1
                else:
                    self.state='HOLD'
                    self.counter = self.SlotTime -1
            else:
                self.counter = self.SlotTime -1
                if Channel.state != 'IDLE':
                    self.state = 'HOLD'
                    self.counter = self.SlotTime -1
        else:
            self.counter -= 1

    def transmit(self, Channel):
        if self.counter ==0:
            if Channel.collision:
                if self.BEB == 7:
                    self.packetManager.fail(time=self.iterCount)
                    self.BEB = 0
                    if not self.packetManager.empty():
                        self.state = 'DIFS'
                        self.counter = self.DIFS - 1
                    else:
                        self.state = 'IDLE'
                else:
                    self.BEB+=1
                    self.BEB = np.min((self.BEB,self.maxBEB))
                    self.state = 'DIFS'
                    self.counter = self.DIFS - 1
            else:
                self.packetSuccess +=1
                self.packetManager.success(time=self.iterCount)
                self.BEB = 0
                if not self.packetManager.empty():
                    self.state ='DIFS'
                    self.counter = self.DIFS- 1
                else:
                    self.state='IDLE'

        else:
            self.counter -= 1

    def hold(self,Channel):
        if self.counter==0:
            if Channel.state =='IDLE':
                self.state = 'DIFS'
                self.counter = self.DIFS - 1
            else:
                self.counter =self.SlotTime - 1
        else:
            self.counter -= 1

    def save(self,filepath=''):
        # np.save(filepath+'/agent_'+str(self.id)+'actionTrack.npy',np.array(self.actionTrack))
        np.save(filepath + '/agent_' + str(self.id) + 'successTrack.npy', np.array(self.successTrack))
        np.save(filepath + '/agent_' + str(self.id) + 'generateTrack.npy', np.array(self.generateTrack))
        np.save(filepath + '/agent_' + str(self.id) + self.categories[self.categorie] + '_successList.npy', np.array(self.packetManager.successList))
        np.save(filepath + '/agent_' + str(self.id) + self.categories[self.categorie] + '_failList.npy', np.array(self.packetManager.failList))
        np.save(filepath + '/agent_' + str(self.id) + self.categories[self.categorie] + '_dropList.npy', np.array(self.packetManager.dropList))