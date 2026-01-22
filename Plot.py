import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import envsettings as env
from config import config
class Plot:
    def __init__(self, path, args):
        self.path = path
        self.args = args
    def plotReward(self):
        data = np.load(self.path + '/ChannelrewardTrack.npy')
        # y = data[:, 1]
        y = data
        average = []
        plt.figure(figsize=(6, 3))
        for i in range(500, len(y)):
            average.append(np.mean(y[i - 500:i]))
        # x = range(len(average))
        x = [env.timestep * 120 * i for i in range(len(average))]
        y = average
        plt.plot(x, y)
        plt.xlabel('Time (s)')
        plt.ylabel('Averaged reward')
        plt.grid(True)
        plt.draw()
        date_str = time.strftime('%Y-%m-%d-%H-%M-%S)')
        PPO_fig_path = (self.args.PPO_fig_path + '/numstep_{}_agent_reward_{}'.
                        format(self.args.train_num_step,date_str))
        PPO_data_path = (self.args.PPO_data_path + '/numstep_{}_agent_reward_{}'.
                         format(self.args.train_num_step,date_str))
        # if not os.path.exists(os.path.dirname(PPO_fig_path)):
        #     os.makedirs(os.path.dirname(PPO_fig_path))
        # if not os.path.exists(os.path.dirname(PPO_data_path)):
        #     os.makedirs(os.path.dirname(PPO_data_path))
        # plt.savefig(PPO_fig_path + '.jpg')
        # df = pd.DataFrame({'steps': x, 'loss': y[0]})
        # df.to_csv(PPO_data_path + '.csv')
        plt.show()
        # plt.close()

    def plotThoughput(self):
        # data = np.load(self.path + '/suceessCount.npy')
        data = np.load(self.path + '/agent_0_successTrack.npy')
        y = data
        throughputs = np.zeros((len(data) + 1, len(data[0])))
        # labels = self.args.agents.copy()
        labels = ['DQN 1', 'DQN 2', 'DQN 3', 'DQN 4', 'DQN 5', 'DQN 6', 'DQN 7', 'PPO 1']
        labels.append('Total')
        for i in range(len(data)):
            for j in range(len(y[0])):
                throughputs[i, j] = (np.sum(y[i,:j]) * 120 / (j + 1e-10))
        throughputs[-1] = np.sum(throughputs[:-1], axis=0)
        plt.figure(figsize=(6, 3))
        x = [env.timestep * i for i in range(len(throughputs[0]))]
        colors = ['r', 'b', 'g', 'y','c','m','k','r','b','g','y','c','m','k']
        linestyles = ['--', '-', '-.', ':','-',':','-.', '-', '-.',':','-',':','-.',':']
        for i in range(len(data) + 1):
            plt.plot(x, throughputs[i], label='Agent{}'.format(i+1), color=colors[i],
                     linewidth=1, linestyle=linestyles[i])
        plt.rcParams.update({'font.size': 8})
        plt.legend(labels, loc='lower right')
        # plt.xlim(0, 3, 0.5)
        plt.ylim(0, 1, 0.2)
        plt.xlabel('Time (s)')
        plt.ylabel('Throughput')
        plt.grid(True)
        plt.draw()
        plt.show()

    def CSMA_delay(self, path, STAs):
        delays = []
        mean_delays_per_STA = []
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'r', 'b', 'g', 'y', 'c', 'm', 'k']
        linestyles = ['--', '-', '-.', ':', '-', ':', '-.', '-', '-.', ':', '-', ':', '-.', ':']
        for STA in STAs:
            data = np.load(path +'/agent_' + str(STA.id) + STA.categories[STA.categorie] + '_successList.npy')
            delay = (data[:,2] - data[:,1]) * env.timestep
            delays.append(delay)
            mean_delays_per_STA.append(np.mean(delay))
        meanDelay = np.mean(np.concatenate(delays))
        delayJitter = np.var(np.concatenate(delays))
        return meanDelay, delayJitter, mean_delays_per_STA

    def OCA_delay(self, path, STAs):
        delays = []
        mean_delays_per_STA = []
        for STA in STAs:
            data = np.load(path +'/agent_' + str(STA.id) + '_successList.npy')
            delay = (data[:,2] - data[:,1]) * env.timestep
            delays.append(delay)
            mean_delays_per_STA.append(np.mean(delay))
        meanDelay = np.mean(np.concatenate(delays))
        delayJitter = np.var(np.concatenate(delays))
        return meanDelay, delayJitter, mean_delays_per_STA

    def CSMA_delay_cdf(self, path, STAs):
        # labels = ['AC_BE 1', 'AC_BE 2', 'AC_BE 3', 'AC_BE 4', 'AC_BE 5', 'AC_BE 6', 'AC_BE 7', 'AC_BE 8', 'AC_BE 9']
        # labels = ['AC_BE 1', 'AC_BE 2', 'AC_BE 3', 'AC_BE 4', 'AC_BE 5', 'AC_BE 6', 'AC_BE 7', 'AC_BE 8', 'AC_BE 9']
        labels = ['AC_VO 1', 'AC_VO 2', 'AC_VO 3', 'AC_VO 4', 'AC_VO 5', 'AC_VO 6', 'AC_VO 7', 'AC_VO 8', 'AC_VO 9']
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'r', 'b', 'g', 'y', 'c', 'm', 'k']
        linestyles = ['--', '-', '-.', ':', '-', ':', '-.', '-', '-.', ':', '-', ':', '-.', ':']
        i = 0
        for STA in STAs:
            data = np.load(path +'/agent_' + str(STA.id) + STA.categories[STA.categorie] + '_successList.npy')
            # 计算数据的累积分布
            delay = (data[:,2] - data[:,1]) * env.timestep
            sorted_delay = np.sort(delay)
            yvals = np.arange(len(sorted_delay)) / float(len(sorted_delay))
            # plt.plot(sorted_delay, yvals)
            plt.plot(sorted_delay, yvals, label=labels[i], color=colors[i],
                     linewidth=2.5, linestyle=linestyles[i])
            i += 1
        # 绘制CDF
        plt.xlabel('Delay (s)')
        plt.ylabel('Probability')
        plt.legend(labels, loc='best')
        # plt.title('Cumulative Distribution Function (CDF)')
        plt.grid(True)
        plt.show()

    def OCA_delay_cdf(self, path, STAs):
        labels = ['DQN 1', 'DQN 2', 'DQN 3', 'DQN 4', 'DQN 5', 'PPO 1', 'PPO 2', 'PPO 3', 'PPO 4']
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'r', 'b', 'g', 'y', 'c', 'm', 'k']
        linestyles = ['--', '-', '-.', ':', '-', ':', '-.', '-', '-.', ':', '-', ':', '-.', ':']
        i = 0
        for STA in STAs:
            OCAdata = np.load(path + '/agent_' + str(STA.id) + '_successList.npy')
            # 计算数据的累积分布
            OCAdelay = (OCAdata[:, 2] - OCAdata[:, 1]) * env.timestep
            OCA_delay = np.sort(OCAdelay)
            max_OCAdelay = OCA_delay[-1]
            OCAextra_points = 200  # 或者你选择的任何数量
            extra_OCAdelay = np.linspace(max_OCAdelay, max_OCAdelay * 2, OCAextra_points)
            OCAdelay = np.append(OCA_delay, extra_OCAdelay)
            OCA_yvals = np.append(np.arange(len(OCAdelay) - OCAextra_points) / float(len(OCAdelay) - OCAextra_points),
                                  [1] * OCAextra_points)
            plt.plot(OCAdelay, OCA_yvals, label=labels[i], color=colors[i],
                      linewidth=2.5, linestyle=linestyles[i])
            # plt.plot(OCAdelay, OCA_yvals)
            i += 1
        # 绘制CDF
        plt.xlabel('Delay (s)')
        plt.ylabel('Probability')
        # plt.title('Cumulative Distribution Function (CDF)')
        plt.legend(labels, loc='best')
        plt.grid(True)
        plt.show()

    def plot_CDF(self, OCApath, CSMApath, STAs):
        labels = ['DQN 1', 'DQN 2', 'PPO 1', 'PPO 2', 'AC_VO 1', 'AC_VO 2', 'AC_VO 3', 'AC_VO 4']
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'r', 'b', 'g', 'y', 'c', 'm', 'k']
        linestyles = ['--', '-', '-.', ':', '-', ':', '-.', '-', '-.', ':', '-', ':', '-.', ':']
        i = 0
        for STA in STAs:
            OCAdata = np.load(OCApath +'/agent_' + str(STA.id) + '_successList.npy')
            # 计算数据的累积分布
            OCAdelay = (OCAdata[:,2] - OCAdata[:,1]) * env.timestep
            OCA_delay = np.sort(OCAdelay)
            max_OCAdelay = OCA_delay[-1]
            OCAextra_points = 200  # 或者你选择的任何数量
            extra_OCAdelay = np.linspace(max_OCAdelay, max_OCAdelay * 2, OCAextra_points)
            OCAdelay = np.append(OCA_delay, extra_OCAdelay)
            OCA_yvals = np.append(np.arange(len(OCAdelay) - OCAextra_points) / float(len(OCAdelay) - OCAextra_points),
                                 [1] * OCAextra_points)
            plt.plot(OCAdelay, OCA_yvals, label='Agent{}'.format(i), color=colors[i],
                     linewidth=2, linestyle=linestyles[i])
            # plt.plot(OCAdelay, OCA_yvals)
            i += 1
        i = len(STAs)
        for STA in STAs:
            CSMAdata = np.load(CSMApath +'/agent_' + str(STA.id) + STA.categories[STA.categorie] + '_successList.npy')
            # 计算数据的累积分布
            CSMAdelay = (CSMAdata[:,2] - CSMAdata[:,1]) * env.timestep
            CSMA_delay = np.sort(CSMAdelay)
            max_CSMA_delay = CSMA_delay[-1]
            CSMAextra_points = 10  # 或者你选择的任何数量
            extra_CSMA_delay = np.linspace(max_CSMA_delay, max_CSMA_delay * 1, CSMAextra_points)
            CSMA_delay = np.append(CSMA_delay, extra_CSMA_delay)
            CSMA_yvals = np.append(np.arange(len(CSMA_delay) - CSMAextra_points) / float(len(CSMA_delay) - CSMAextra_points),
                [1] * CSMAextra_points)
            plt.plot(CSMA_delay, CSMA_yvals, label='CSMA{}'.format(i), color=colors[i],
                     linewidth=2, linestyle=linestyles[i])
            # plt.plot(CSMA_delay, CSMA_yvals)
            i += 1
        # 绘制CDF
        plt.xlabel('Delay (s)')
        plt.ylabel('Probability')
        plt.legend(labels, loc='best')
        # plt.title('Cumulative Distribution Function (CDF)')
        plt.grid(True)
        plt.show()

    def plot_unsaturated_traffic(self, OCApath, CSMApath, STAs):
        OCAdelay = []
        CSMAdelay = []
        for STA in STAs:
            OCAdata = np.load(OCApath + '/agent_' + str(STA.id) + '_successList.npy')
            # 计算数据的累积分布
            OCA_delay = (OCAdata[:, 2] - OCAdata[:, 1]) * env.timestep
            OCA_delay = np.sort(OCA_delay[0:1140])
            OCAdelay.append(OCA_delay)
            CSMAdata = np.load(CSMApath + '/agent_' + str(STA.id) + STA.categories[STA.categorie] + '_successList.npy')
            CSMA_delay = (CSMAdata[:, 2] - CSMAdata[:, 1]) * env.timestep
            CSMA_delay = np.sort(CSMA_delay)
            CSMAdelay.append(CSMA_delay[:780])
        OCAdelay = np.mean(np.array(OCAdelay), axis=0)
        CSMA_delay = np.mean(np.array(CSMAdelay), axis=0)

        # 确定最大延迟值
        max_OCAdelay = OCAdelay[-1]
        max_CSMA_delay = CSMA_delay[-1]

        # 决定在100%之后要追加多少个点
        OCAextra_points = 200  # 或者你选择的任何数量
        CSMAextra_points = 10  # 或者你选择的任何数量
        # 生成附加的延迟值，例如从最大值开始递增
        extra_OCAdelay = np.linspace(max_OCAdelay, max_OCAdelay * 3, OCAextra_points)
        extra_CSMA_delay = np.linspace(max_CSMA_delay, max_CSMA_delay * 1.5, CSMAextra_points)

        # 将额外的延迟值添加到原始数组中
        OCAdelay = np.append(OCAdelay, extra_OCAdelay)
        CSMA_delay = np.append(CSMA_delay, extra_CSMA_delay)

        # 对于这些额外的点，累积概率都保持为1
        OCA_yvals = np.append(np.arange(len(OCAdelay) - OCAextra_points) / float(len(OCAdelay) - OCAextra_points),
                              [1] * OCAextra_points)
        CSMA_yvals = np.append(np.arange(len(CSMA_delay) - CSMAextra_points) / float(len(CSMA_delay) - CSMAextra_points),
                               [1] * CSMAextra_points)

        plt.plot(OCAdelay, OCA_yvals,)
        plt.plot(CSMA_delay, CSMA_yvals,)
        plt.xlabel('Delay (s)')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.show()
if __name__ == '__main__':
    args = config(mode=True)
    currentPath = args.dir + '/numOCA={}'.format(8)
    plot = Plot(currentPath, args)
    #plot.plotReward()
    plot.plotThoughput()