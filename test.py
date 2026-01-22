import numpy as np
import random
from config import config
import matplotlib.pyplot as plt
preSlot2lastTransmit = [0,2,2]
preSlot2lastTransmit = np.array(preSlot2lastTransmit)
actions = [0,0,1]
actions = np.array(actions)
if np.argmax(actions) in np.arange(#the agent with max d success
         len(preSlot2lastTransmit[preSlot2lastTransmit == np.max(preSlot2lastTransmit)])):
    reward = 1
else:
    reward = -1
print(reward)
#
# if np.argmax(actions) in np.argwhere(preSlot2lastTransmit==np.amax(preSlot2lastTransmit)):
#     reward = 1
# else:
#     reward = -1
# print(reward)
# arg = config(False)
# path = arg.dir + '/numOCA=2'
# data = np.load(path + '/suceessCount.npy')
# y1 = data[0]
# y2 = data[1]
# average = []
# throughtput1 = []
# throughtput2 = []
# for i in range(len(y1)):
#     throughtput1.append(np.sum(y1[:i]) * 120 / (i + 1e-10))
#
# for i in range(len(y2)):
#     throughtput2.append(np.sum(y2[:i]) * 120 / (i + 1e-10))
# plt.figure()
# x = range(len(throughtput1))
# labels= ['Agent1','Agent2']
# plt.plot(x, throughtput1, label='Agent1',color='r',linewidth=1,linestyle='--')
# plt.plot(x, throughtput2, label='Agent2',color='b',linewidth=1,linestyle='-')
# plt.legend(labels, loc='best')
# plt.xlabel('Time step')
# plt.ylabel('Throughput')
# plt.draw()
# plt.show()
a = np.amax(preSlot2lastTransmit)
b = np.argwhere(preSlot2lastTransmit == np.amax(preSlot2lastTransmit))
if np.argmax(actions) in np.argwhere(preSlot2lastTransmit == np.amax(preSlot2lastTransmit)):
    reward = 1