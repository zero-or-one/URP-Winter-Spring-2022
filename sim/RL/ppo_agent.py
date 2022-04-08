# -- coding: utf-8 --
from typing import Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch.distributions import Categorical
from utils import seed_all

SPEED_PARAM = 0.3

class PPOAgent:
    def __init__(self, env, lr_actor, lr_critic,
                 gamma=0.99, K_epoch=3, eps_clip=0.2,
                 gae_lambda=0.95, conf=None):
        self.config = conf
        self.env = env
        if torch.cuda.is_available():
            device = 'cuda:0'
            torch.cuda.empty_cache()
            print('Device set to', torch.cuda.get_device_name(torch.device(device)))
        else:
            device = 'cpu'
            print('Device set to cpu')
        self.device = device
        self.gamma = gamma
        self.K_epoch = K_epoch
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.buffer = Buffer()

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.actor_old = Actor(state_dim, action_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())


    def train(self, episode_num, update_interval, render_interval, save_interval, eval_episodes=10, initial_ep=None):
        total_timestep = 0 if initial_ep == None else initial_ep
        scores = []
        actor_losses, critic_losses = [], []
        best_score = -np.inf
        print('Starting training from', total_timestep)
        for i_epi in range(episode_num):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                for i in range(update_interval):
                    total_timestep += 1
                    prob = self.actor_old(torch.tensor([state], dtype=torch.float)).to(self.device)
                    m = Categorical(prob)
                    action = m.sample().item()
                    state_next, reward, done, info = self.env.step(action)
                    reward *= (1 + SPEED_PARAM * self.env.speed)
                    self.env.render()
                    self.put_data(*(state, action, reward / 100, state_next, prob[0][action].item(), done))
                    state = state_next
                    score += reward
                    if total_timestep % render_interval == 0:
                        print("# of episode :{}, last 10 ep avg score : {:.1f}".format(i_epi, np.mean(scores[-10:])))
                        self._plot(total_timestep, scores, actor_losses, critic_losses)
                        cur_score = self.eval(reload_model=False, render=True, episode_num=eval_episodes)
                        print(f'test for {eval_episodes} episodes, mean score{cur_score}')
                    if total_timestep % save_interval == 0:
                        self.save(f'step_{total_timestep}')
                        cur_score = self.eval(reload_model=False, render=False, episode_num=eval_episodes)
                        if cur_score >= best_score or cur_score > 10:
                            self.save(f'bestmodel_step{total_timestep}')
                            best_score = cur_score
                    if done:
                        scores.append(score)
                        break
                actor_loss, critic_loss = self.update()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

    def eval(self, reload_model=False, render=False, model=None, episode_num=5):
        #print("Evaluating")
        if reload_model:
            self.load(model)
        scores = []
        for i_epi in range(episode_num):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                if render:
                    self.env.render()
                with torch.no_grad():
                    prob = self.actor_old(torch.tensor([state], dtype=torch.float)).to(self.device)
                    action = torch.argmax(prob).item()
                    state_next, reward, done, info = self.env.step(action)
                    reward *= (1 + SPEED_PARAM * self.env.speed)
                    score += reward
                    state = state_next
            scores.append(score)
        self.env.close()
        #print("Finished evaluating")
        return np.mean(scores)

    def update(self):
        state, action, reward, state_next, prob_action, done_mask = self.buffer.get_batch()
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        state_next = torch.tensor(state_next, dtype=torch.float).to(self.device)
        prob_action = torch.tensor(prob_action).to(self.device)
        done_mask = torch.tensor(done_mask).to(self.device)
        actor_loss_sum, critic_loss_sum = 0, 0
        for i in range(self.K_epoch):
            td_target = reward + self.gamma * torch.squeeze(self.critic(state_next), dim=-1) * done_mask
            delta = (td_target - self.critic(state)).cpu().detach().numpy()
            advantage_list = []
            advantage = 0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.gae_lambda * advantage + delta_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

            pi = self.actor(state)
            pi_action = pi.gather(1, action)
            ratio = torch.exp(torch.log(pi_action) - torch.log(prob_action))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            critic_loss = F.smooth_l1_loss(self.critic(state), td_target.detach())
            actor_loss = -torch.min(surr1, surr2).mean()
            loss = actor_loss + critic_loss
            actor_loss_sum += actor_loss.item()
            critic_loss += critic_loss.item()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer.clear()

        return actor_loss_sum/self.K_epoch, critic_loss_sum/self.K_epoch


    def put_data(self, *transition):
        self.buffer.put_data(*transition)

    def clip_grad(self):
        for param in self.actor.parameters():
            clip_grad_norm(param, 10)
        for param in self.critic.parameters():
            clip_grad_norm(param)

    def _plot(self, total_timestep, scores, actor_losses, critic_losses):
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (total_timestep, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('frame %s. actor_loss: %s' % (total_timestep, np.mean(actor_losses[-10:])))
        plt.plot(actor_losses)
        plt.subplot(133)
        plt.title('frame %s. critic_loss: %s' % (total_timestep, np.mean(actor_losses[-10:])))
        plt.plot(critic_losses)
        plt.show()

    def save(self, filename):
        if self.config:
            model_path = self.config.model_dir
            torch.save(self.actor_old.state_dict(), model_path + f'/{filename}_actor.pkl')
            torch.save(self.critic.state_dict(), model_path + f'/{filename}_critic.pkl')

    def load(self, filename):
        model_path = self.config.model_dir
        state_dict_actor = torch.load(model_path + f'/{filename}_actor.pkl', map_location=lambda storage, loc: storage)
        self.actor_old.load_state_dict(state_dict_actor)
        self.actor.load_state_dict(state_dict_actor)
        state_dict_critic = torch.load(model_path + f'/{filename}_critic.pkl', map_location=lambda storage, loc: storage)
        self.critic.load_state_dict(state_dict_critic)



class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Actor, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, input_):
        action_prop = F.softmax(self.nn(input_), dim=1)
        return action_prop


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(Critic, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_):
        value = self.nn(input_)
        return value


class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.log_probs[:]
        del self.is_terminals[:]

    def put_data(self, *transition):
        state, action, reward, next_state, prob, done = transition
        self.states.append(state)
        self.actions.append([action])
        self.rewards.append([reward])
        self.next_states.append([next_state])
        self.log_probs.append([prob])
        self.is_terminals.append([int(not done)])

    def get_batch(self):
        return self.states, self.actions, self.rewards, \
    self.next_states, self.log_probs, self.is_terminals



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    seed_all(env)
    model = PPOAgent(env=env, lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epoch=3,
                     eps_clip=0.1)
    score = 0.0
    print_interval = 20
    T_horizon = 20
    episode_num = 1000

    model.train(episode_num=episode_num, update_interval=20, render_interval=2000, save_interval=3000)

    for n_epi in range(10000):
        s = env.reset()
        done = False
        i = 0
        while not done:
            for t in range(T_horizon):
                i += 1
                prob = model.actor_old(torch.tensor([s], dtype=torch.float).to(model.device))
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                model.put_data(*(s, a, r/100, s_prime, prob[0][a].item(), done))
                s = s_prime
                score += r
                # if done or i == 20000:
                if done:
                    # print(f'done_{i}_{r}')
                    # done = True
                    # i = 0
                    break
            model.update()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            # if score / print_interval > -1000:
            #     env = gym.make('MountainCar-v0')
            score = 0.0

    env.close()