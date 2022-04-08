# -- coding: utf-8 --

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import collections
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from segment_tree import MinSegmentTree, SumSegmentTree
from torch.nn.utils import clip_grad_norm_

class DQN_MARL:
    def __init__(self, env, agents):
        self.agent_num = len(agents)
        self.agents = agents

    def train(self, num_steps, render_interval, save_interval=100000, update_interval=-1, eval_episodes=10, initial_ep=None):
        self.is_test = False
        obs = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        best_score = -np.inf
        if initial_ep == None:
            initial_ep = 0
        print('Starting training from', initial_ep)
        for i_step in range(initial_ep, num_steps+1):
            actions = []
            for agent in self.agents:
                action = agent.select_action(obs)
                actions.append(action)
            obs, reward, done, info = self.env.step(actions)
            score += reward

            if done:
                scores.append(score)
                score = 0
                if len(scores) > 20:
                    self.update_epsilons()
                epsilons.append(self.agent[0].epsilon)
                obs = self.env.reset()

            # backprop
            for agent in self.agents:
                l = []
                if len(agent.memory) >= agent.batch_size:
                    samples = agent.memory.sampe_batch()
                    loss = agent._compute_dqn_loss(samples)
                    agent.optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(agent.dqn.parameters(), 10.0)
                    agent.optimizer.step()
                    l.append(loss.item())
                    update_cnt += 1

                    # copy Q to target Q
                    if update_cnt % agent.target_update:
                        agent._target_hard_update()
                losses.append(l)

            # save
            if i_step % save_interval == 0:
                for i in range(self.agent_num):
                    self.agents[i].save(f'step_{i_step}_agent{i}')
                    cur_score = self.agents[i].eval(reload_model=False, render=False, num_episodes=eval_episodes)
                    if cur_score >= best_score or cur_score >= 8: # limit is updatable
                        self.save(f'best_step_{i_step}_agent{i}')
                        best_score = cur_score
                    self.is_test = False

            # plot
            if i_step % render_interval == 0:
                self._plot(i_step, scores, losses[0], epsilons) # single agent is enough
                print('episode', len(scores), 'steps', i_step)
                print('Score', np.mean(np.mean(scores[-10:])))

            self.env.close()

    def eval(self, reload_model=False, render=False, model=None, num_episodes=5):
        for agent in self.agents:
            agent.is_test = True
            if reload_model:
                agent.load(model)
        scores = []
        for i in range(num_episodes):
            obs = self.env.reset()
            done = False
            score = 0
            while not done:
                obs, reward, done = self.eval_step(obs)
                score += reward
                if render:
                    self.env.render()
            self.env.close()
            scores.append(score)
        return np.mean(scores)

    def step(self, actions):
        new_obs, reward, done, _ = self.env.step(actions)
        for agent in self.agents:
            self.agent.transition += [reward, new_obs, done]
            self.agent.memory.store(*self.transition)
        return new_obs, reward, done

    def eval_step(self, obs):
        actions = []
        for agent in self.agents:
            action = self.dqn(
                torch.FloatTensor(obs).to(self.device)
            ).argmax().detach().cpu().numpy()
            actions.append(action)
        new_obs, reward, done, _ = self.env.step(action)
        return new_obs, reward, done


class PPO_MARL:
    def __init__(self, env, agents):
        self.agent_num = len(agents)
        self.agents = agents

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
                    score += reward
                    state = state_next
            scores.append(score)
        self.env.close()
        #print("Finished evaluating")
        return np.mean(scores)