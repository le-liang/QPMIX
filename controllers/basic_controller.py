import numpy as np
import sys
from modules.mixer.qimx import QMixer
from modules.agents import REGISTRY as agent_REGISTRY
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from os.path import dirname
import torch.nn as nn
import copy
from torch.optim import RMSprop
from torch.optim import Adam
from collections import namedtuple
import random
import envsettings as env
from config import config
import torch
args = config(False)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# This multi-agent controller
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'tot_reward','ind_reward', 'next_state', 's_t', 's_t_plus_1'])
class BasicMAC:
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.args = args
        self._build_agents()
        # self.agent_output_type = args.agent_output_type

        #self.action_selector = action_REGISTRY[args.action_selector](args)
        self.mixer = QMixer(args).to(self.args.device)
        self.hidden_states = None
        self.target_mixer = copy.deepcopy(self.mixer)
        self.loss_func = nn.MSELoss()
        self.qmix_params = self.qmix_parameters()
        self.qmix_params += list(self.mixer.parameters())
        self.qmix_optimiser = RMSprop(params=self.qmix_params, lr=args.qmix_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.qmix_loss = []
        self.V_params = self.mixer.V.parameters()
        self.V_optimiser = Adam(self.V_params, lr=self.args.PPO_V_lr)
        self.V_loss = []
        self.q_individual_params = self.q_individual_params()
        self.q_optimiser = RMSprop(params=self.q_individual_params, lr=args.qmix_lr, alpha=args.optim_alpha,
                                      eps=args.optim_eps)
        self.entropy = self.args.entropy
        self.ppo_train_num = 0
    # def select_actions(self, observations, epsi=0.):
    #     agent_outputs,  action_probs = self.forward(observations, epsi)
    #     return agent_outputs, action_probs
    #
    # def forward(self, observations, epsi=0.):
    #     agent_outputs = []
    #     action_probs = []
    #     if self.args.isSharePara:
    #         if 'DQN' in self.args.agent and 'PPO' in self.args.agent:
    #                 agent_outs = self.agents[0].choose_action(observations[:self.args.num_dqn], epsi)
    #                 agent_outputs.append(agent_outs)
    #                 agent_outs, action_prob= self.agents[1].choose_action(observations[self.args.num_dqn:], epsi)
    #                 action_probs.append(action_prob)
    #                 agent_outputs.append(agent_outs)
    #         elif 'DQN' in self.args.agent:
    #             agent_outs = self.agents[0].choose_action(observations, epsi)
    #             agent_outputs.append(agent_outs)
    #         elif 'PPO' in self.args.agent:
    #             agent_outs, action_prob = self.agents[0].choose_action(observations[self.args.num_dqn:], epsi)
    #             action_probs.append(action_prob)
    #             agent_outputs.append(agent_outs)
    #         return np.array(agent_outputs), np.array(action_probs)
    #     else:
    #         for i in range(self.n_agents):
    #             if self.agents[i].agent_type == 'DQN':
    #                 agent_outs = self.agents[i].choose_action(observations[i], epsi)
    #                 agent_outputs.append(agent_outs)
    #             if self.agents[i].agent_type == 'PPO':
    #                 agent_outs, action_prob = self.agents[i].choose_action(observations[i])
    #                 agent_outputs.append(agent_outs)
    #                 action_probs.append(action_prob)
    #         return np.array(agent_outputs), np.array(action_probs)

    def choose_action(self, observation, id, epsi):
        if self.agents[id].agent_type == 'DQN':
            agent_outs = self.agents[id].choose_action(observation, epsi)
            return agent_outs
        elif self.agents[id].agent_type == 'PPO':
            agent_outs, action_prob = self.agents[id].choose_action(observation, epsi)
            return agent_outs, action_prob

    def DQN_train(self):
        # batch_o_ts = np.zeros([self.n_agents, self.args.DQN_batch_size, self.args.input_shape])
        # batch_o_ts_plus_1 = np.zeros_like(batch_o_ts)
        # batch_actions = np.zeros([self.n_agents, self.args.DQN_batch_size, 1])
        batch_ind_rewards = torch.zeros([self.args.DQN_batch_size,self.n_agents,1]).to(self.args.device)
        batch_tot_rewards = torch.zeros([self.args.DQN_batch_size, 1]).to(self.args.device)
        # q_values = []
        chosen_action_qvals = torch.zeros([self.args.DQN_batch_size,self.n_agents, 1]).to(self.args.device)
        target_max_qvals = torch.zeros([self.args.DQN_batch_size, self.n_agents, 1]).to(self.args.device)

        indexes = random.sample(range(0, self.agents[0].DQNMemory.count), self.args.DQN_batch_size)
        for i in range(self.n_agents):
            if self.agents[i].agent_type == 'DQN':
                batch_o_t, batch_o_t_plus_1, batch_action, tot_rewards,ind_rewards, s_t, s_t_plus_1 = (
                    self.agents[i].DQNMemory.sample(indexes))
            if self.agents[i].agent_type == 'PPO':
                batch_o_t, batch_o_t_plus_1, batch_action, tot_rewards,ind_rewards, s_t, s_t_plus_1 = (
                    self.agents[i].DQNMemory.sample(indexes))
            action = torch.LongTensor(batch_action).to(self.args.device)
            tot_rewards = torch.FloatTensor(tot_rewards).to(self.args.device)
            ind_rewards = torch.FloatTensor(ind_rewards).to(self.args.device)
            batch_o_t = torch.FloatTensor(np.float32(batch_o_t)).to(self.args.device)
            batch_o_t_plus_1 = torch.FloatTensor(np.float32(batch_o_t_plus_1)).to(self.args.device)
            s_t = torch.FloatTensor(np.float32(s_t)).to(self.args.device)
            s_t_plus_1 = torch.FloatTensor(np.float32(s_t_plus_1)).to(self.args.device)
            # batch_o_ts[i] = o_t
            # batch_o_ts_plus_1[i] = o_t_plus_1
            # batch_actions[i] = action
            batch_ind_rewards[:,i] = ind_rewards
            batch_tot_rewards = tot_rewards
            q_value = self.agents[i].model(batch_o_t)
            chosen_action_qvals[:,i] = q_value.gather(dim=1, index=action)
            if self.args.double_q:
                next_action = self.agents[i].model(batch_o_t_plus_1).max(1)[1].unsqueeze(1)
                next_q_values = self.agents[i].target_model(batch_o_t_plus_1)
                next_q_value = next_q_values.gather(dim=1, index=next_action)
            else:
                next_q_value = self.agents[i].target_model(batch_o_t_plus_1).max(1)[0].unsqueeze(1)
            target_max_qvals[:,i] = next_q_value
        mix_q, q_ind, v = self.mixer(chosen_action_qvals, s_t)
        target_mix_q, target_q_ind, v = self.target_mixer(target_max_qvals,s_t_plus_1)
        targets_mix = batch_tot_rewards + self.args.discount * target_mix_q
        target_q_ind = batch_ind_rewards + self.args.discount * target_q_ind
        target_q_ind = batch_ind_rewards + self.args.discount * target_max_qvals
        td_error = self.loss_func(mix_q, targets_mix.detach()) + self.loss_func(chosen_action_qvals, target_q_ind.detach())
        self.qmix_optimiser.zero_grad()
        self.q_optimiser.zero_grad()
        td_error.backward()
        nn.utils.clip_grad_norm_(self.qmix_params, self.args.qmix_max_grad_norm)
        nn.utils.clip_grad_norm_(self.q_individual_params, self.args.qmix_max_grad_norm)
        self.qmix_optimiser.step()
        self.q_optimiser.step()

    def PPO_train(self):
        self.entropy = max(0.5 - self.ppo_train_num * (1 - 0.01) / (env.simulationTime/(env.timestep * 120 * 20) - 1), 0.0)
        self.ppo_train_num += 1
        for i in range(self.n_agents):
            if self.agents[i].agent_type == 'PPO':
                o_t, o_t_plus_1, a, r_tot, r_ind, s_t, s_t_plus_1, a_old_prob = self.agents[i].PPOMemory.sample()
                for k in range(self.args.K_epoch):
                    with torch.no_grad():
                        td_target = r_tot + self.args.discount * self.target_mixer.V(s_t_plus_1)
                        td_error = r_tot + self.args.discount * self.mixer.V(s_t_plus_1) - self.mixer.V(s_t)
                        td_error = td_error.detach().cpu().numpy()
                        advantage = []  # Advantage Function
                        adv = 0.0
                        for td in td_error[::-1]:
                            adv = adv * self.args.LAMBDA * self.args.discount + td[0]
                            advantage.append(adv)
                        advantage.reverse()
                        # advantage = self.agents[i].model(o_t).gather(dim=1, index=a) - self.mixer.V(s_t)
                        advantage = torch.tensor(advantage, dtype=torch.float).reshape(-1, 1).to(self.args.device)
                        # Trick: Normalization
                        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-7)
                    dist_now_entropy = Categorical(probs=self.agents[i].actor(o_t)).entropy()
                    a_new_prob = self.agents[i].actor(o_t).gather(1, a)
                    ratio = a_new_prob / a_old_prob.detach()
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.args.clip_param, 1 + self.args.clip_param) * advantage

                    actor_loss = (- torch.min(surr1, surr2)-self.entropy* dist_now_entropy).mean()
                    # actor_loss = (- torch.min(surr1, surr2)).mean()
                    self.agents[i].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.agents[i].actor_loss.append(actor_loss.item())
                    nn.utils.clip_grad_norm_(self.agents[i].actor.parameters(), self.args.PPO_max_grad_norm)
                    self.agents[i].actor_optimizer.step()

                    critic_loss = self.loss_func(td_target.detach(), self.mixer.V(s_t))
                    self.V_optimiser.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.V_params, self.args.PPO_max_grad_norm)
                    self.V_optimiser.step()
                self.agents[i].old_actor.load_state_dict(self.agents[i].actor.state_dict())
        self.target_mixer.V.load_state_dict(self.mixer.V.state_dict())
    def train(self):
        self.DQN_train()
        self.PPO_train()

    def insert(self, obs_old, obs, action, rewards, s_t, s_t_plus_1, actions_prob):
        if self.args.isSharePara:
            for i in range(self.args.n_agents):
                self.agents[i].DQNMemory.add(obs_old[i], obs[i], rewards[i], rewards[-1], action[i], s_t, s_t_plus_1)
                if self.agents[i].agent_type == 'PPO':
                    # trans = Transition(obs_old[i], action[i], actions_prob[i-self.args.num_dqn],
                    #                    self.args.gamma*rewards[i]+rewards[-1], s_t)
                    trans = Transition(obs_old[i], action[i], actions_prob[i - self.args.num_dqn],
                                       rewards[-1], obs[i], s_t, s_t_plus_1)
                    self.agents[i].PPOMemory.add(trans)
        else:
            for i in range(self.args.n_agents):
                self.agents[i].DQNMemory.add(obs_old[i], obs[i], rewards[i], rewards[-1], action[i], s_t, s_t_plus_1)
                if self.agents[i].agent_type == 'PPO':
                    # trans = Transition(obs_old[i], action[i], actions_prob[i - self.args.num_dqn],
                    #                    rewards[i], obs[i], s_t, s_t_plus_1)
                    # trans = Transition(obs_old[i], action[i], actions_prob[i - self.args.num_dqn],
                                       # rewards[i], obs[i], s_t, s_t_plus_1)
                    trans = Transition(obs_old[i], action[i], actions_prob[i],
                                                            rewards[-1], rewards[i], obs[i], s_t, s_t_plus_1)
                    self.agents[i].PPOMemory.add(trans)

    def update_target_network(self):
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        for i in range(self.n_agents):
            self.agents[i].target_model.load_state_dict(self.agents[i].model.state_dict())

    # def init_hidden(self, batch_size):
    #     self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def qmix_parameters(self):
        param = list()
        # Do not update the actor parameter in PPO
        for i in range(self.n_agents):
            param += list(self.agents[i].model.parameters())
            # if self.agents[i].agent_type=='PPO':
            #     param += list(self.agents[i].actor.parameters())
        return param

    def q_individual_params(self):
        param = list()
        for i in range(self.n_agents):
            param += list(self.agents[i].model.parameters())
        return param
    # def V_parameters(self):
    #     param = list(self.mixer.V.parameters())
    #     return param

    # def load_state(self, other_mac):
    #     self.agent.load_state_dict(other_mac.agent.state_dict())

    # def cuda(self):
    #     self.agent.cuda()

    def save_models(self, currentPath):
        # torch.save(self.agent.state_dict(), "{}/agent.th".format(path))
        for i in range(self.n_agents):
            self.agents[i].save_models(i, currentPath)
        #current_dir = dirname(dirname(os.path.realpath(__file__)))
        model_path = os.path.join(currentPath, '\heterogeneous'+str(self.n_agents)+'/mixDQN' +
                                  str(self.args.num_dqn)+ '/mixing')
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.mixer.state_dict(), model_path + '_a.ckpt')
        torch.save(self.target_mixer.state_dict(), model_path + '_c.ckpt')
    def load_models(self, currentPath):
        for i in range(self.n_agents):
            self.agents[i].load_models(i, currentPath)

    def _build_agents(self):
        self.agents = []
        if self.args.isSharePara:
            if 'DQN' in self.args.agents:
                agent = agent_REGISTRY['DQN'](self.args)
                self.agents.append(agent)
            if 'PPO' in self.args.agents:
                agent = agent_REGISTRY['PPO'](self.args)
                self.agents.append(agent)
        else:
            for i in range(self.n_agents):
                agent = agent_REGISTRY[self.args.agents[i]](self.args)
                self.agents.append(agent)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, arg):
        return arg.input_shape

    def save_figure(self, record_reward):
        self.save_loss()
        self.save_reward(record_reward)

    def save_loss(self):
        # QMIX loss
        x = range(len(self.qmix_loss))
        y = self.qmix_loss
        plt.plot(x, y)
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.draw()
        # plt.show()
        date_str = time.strftime('%Y-%m-%d-%H-%M-%S)')
        DQN_fig_path = (self.args.DQN_fig_path + '/numstep_{}_qmix_loss_{}'.
                        format(self.args.train_num_step,date_str))
        DQN_data_path = (self.args.DQN_data_path + '/numstep_{}_qmix_loss_{}'.
                         format(self.args.train_num_step,date_str))
        if not os.path.exists(os.path.dirname(DQN_fig_path)):
            os.makedirs(os.path.dirname(DQN_fig_path))
        if not os.path.exists(os.path.dirname(DQN_data_path)):
            os.makedirs(os.path.dirname(DQN_data_path))
        plt.savefig(DQN_fig_path + '.jpg')
        df = pd.DataFrame({'steps': x, 'loss': y})
        df.to_csv(DQN_data_path + '.csv')
        plt.close()

        # V loss
        x = range(len(self.V_loss))
        y = self.V_loss
        plt.plot(x, y)
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.draw()
        # plt.show()
        date_str = time.strftime('%Y-%m-%d-%H-%M-%S)')
        PPO_fig_path = (self.args.PPO_fig_path + '/numstep_{}_V_loss_{}'.
                        format(self.args.train_num_step, date_str))
        PPO_data_path = (self.args.PPO_data_path + '/numstep_{}_V_loss_{}'.
                         format(self.args.train_num_step, date_str))
        if not os.path.exists(os.path.dirname(PPO_fig_path)):
            os.makedirs(os.path.dirname(PPO_fig_path))
        if not os.path.exists(os.path.dirname(PPO_data_path)):
            os.makedirs(os.path.dirname(PPO_data_path))
        plt.savefig(PPO_fig_path + '.jpg')
        df = pd.DataFrame({'steps': x, 'loss': y})
        df.to_csv(PPO_data_path + '.csv')
        plt.close()

        for i in range((self.n_agents)):
            if self.agents[i].agent_type == 'PPO':
                x = range(len(self.agents[i].actor_loss))
                y = self.agents[i].actor_loss
                plt.plot(x, y)
                plt.xlabel('steps')
                plt.ylabel('loss')
                plt.draw()
                # plt.show()
                date_str = time.strftime('%Y-%m-%d-%H-%M-%S)')
                PPO_fig_path = (self.args.PPO_fig_path + '/numstep_{}_agent_{}_actor_loss_{}'.
                                format(self.args.train_num_step,i,date_str))
                PPO_data_path = (self.args.PPO_data_path + '/numstep_{}_agent_{}_actor_loss_{}'.
                                 format(self.args.train_num_step,i,date_str))
                if not os.path.exists(os.path.dirname(PPO_fig_path)):
                    os.makedirs(os.path.dirname(PPO_fig_path))
                if not os.path.exists(os.path.dirname(PPO_data_path)):
                    os.makedirs(os.path.dirname(PPO_data_path))
                plt.savefig(PPO_fig_path + '.jpg')
                df = pd.DataFrame({'steps': x, 'loss': y})
                df.to_csv(PPO_data_path + '.csv')
                plt.close()
    def save_reward(self, record_reward):
        x = range(len(record_reward))
        y = record_reward/6
        # y = []
        # for i in range(len(record_reward)):
        #     if i < 500:
        #         a = record_reward[:i+1]
        #     else:
        #         a = record_reward[i-500:i]
        #     y.append(a.mean())

        plt.plot(x, y)
        plt.xlabel('episode')
        plt.ylabel('Return')
        plt.draw()
        date_str = time.strftime('%Y-%m-%d-%H-%M-%S)')
        PPO_fig_path = (self.args.PPO_fig_path + '/numstep_{}_agent_reward_{}'.
                        format(self.args.train_num_step,date_str))
        PPO_data_path = (self.args.PPO_data_path + '/numstep_{}_agent_reward_{}'.
                         format(self.args.train_num_step,date_str))
        if not os.path.exists(os.path.dirname(PPO_fig_path)):
            os.makedirs(os.path.dirname(PPO_fig_path))
        if not os.path.exists(os.path.dirname(PPO_data_path)):
            os.makedirs(os.path.dirname(PPO_data_path))
        plt.savefig(PPO_fig_path + '.jpg')
        df = pd.DataFrame({'steps': x, 'loss': y[0]})
        df.to_csv(PPO_data_path + '.csv')
        plt.close()