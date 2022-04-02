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
from IPython.display import clear_output
import os
import json
from argparse import Namespace
from collections import OrderedDict
from typing import Dict, List, Tuple
from utils import seed_all


class D3QNAgent:
    def __init__(self,
                 env,
                 memory_size,
                 batch_size,
                 target_update,
                 epsilon_decay,
                 max_epsilon = 0.8,
                 min_epsilon = 0.1,
                 gamma = 0.99,
                 beta = 0.6,
                 prior_eps = 1e-6,
                 conf = None
                 ):
        self.config = conf
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.reduc_epsilon = 0.05
        self.target_update = target_update
        self.gamma = gamma
        self.beta = beta
        self.prior_eps = prior_eps

        self.env = env
        self.memory = MemoryBuffer(self.obs_dim, memory_size, batch_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('DEVICE IS ', self.device)

        self.dqn = DuelNN(self.obs_dim, self.action_dim).to(self.device)
        self.dqn_target = DuelNN(self.obs_dim, self.action_dim).to(self.device)
        self.dqn.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters())
        self.transition = list()
        self.is_test = False

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
            action = self.select_action(obs)
            obs, reward, done, info = self.env.step(action)
            score += reward

            if done:
                scores.append(score)
                score = 0
                if len(scores) > 20:
                    self.update_epsilons()
                epsilons.append(self.epsilon)
                obs = self.env.reset()

            # backprop
            if len(self.memory) >= self.batch_size:
                samples = self.memory.sampe_batch()
                loss = self._compute_dqn_loss(samples)
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.dqn.parameters(), 10.0)
                self.optimizer.step()
                losses.append(loss.item())
                update_cnt += 1

                # copy Q to target Q
                if update_cnt % self.target_update:
                    self._target_hard_update()

            # save
            if i_step % save_interval == 0:
                self.save(f'step_{i_step}')
                cur_score = self.eval(reload_model=False, render=False, num_episodes=eval_episodes)
                if cur_score >= best_score or cur_score >= 8: # limit is updatable
                    self.save(f'best_step_{i_step}')
                    best_score = cur_score
                self.is_test = False

            # plot
            if i_step % render_interval == 0:
                self._plot(i_step, scores, losses, epsilons)
                print('episode', len(scores), 'steps', i_step)
                print('Score', np.mean(np.mean(scores[-10:])))

            self.env.close()

    def eval(self, reload_model=False, render=False, model=None, num_episodes=5):
        self.is_test = True
        if reload_model:
            self.load(model)
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

    def select_action(self, obs):
        if self.epsilon > np.random.random():
            action = self.env.action_space.sample()
        else:
            action = self.dqn(
                torch.FloatTensor(obs).to(self.device)
            ).argmax().detach().cpu().numpy()
        self.transition = [obs, action]
        return action

    def step(self, action):
        new_obs, reward, done, _ = self.env.step(action)
        self.transition += [reward, new_obs, done]
        self.memory.store(*self.transition)
        return new_obs, reward, done

    def eval_step(self, obs):
        action = self.dqn(
            torch.FloatTensor(obs).to(self.device)
        ).argmax().detach().cpu().numpy()
        new_obs, reward, done, _ = self.env.step(action)
        return new_obs, reward, done

    def _compute_dqn_loss(self, samples, elementwise=False):
        device = self.device
        state = torch.FloatTensor(samples['obs']).to(device)
        next_state = torch.FloatTensor(samples['next_obs']).to(device)
        action = torch.LongTensor(samples["act"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rew"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        q_value = self.dqn(state).gather(1, action)
        action_max = self.dqn(next_state).argmax(dim=1, keepdim=True)
        next_q_value = self.dqn_target(next_state).gather(1, action_max).detach()
        target = (reward + self.gamma * next_q_value * (1 - done)).to(device)

        if elementwise:
            loss = F.smooth_l1_loss(q_value, target, reduction='none')
        else:
            loss = F.smooth_l1_loss(q_value, target)
        return loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def update_epsilons(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.min_epsilon)

    def save(self, name):
        if self.config:
            model_path = self.config.model_dir
            torch.save(self.dqn.state_dict(), model_path + f'/{name}')

    def load(self, name):
        path_ = self.config.model_dir
        state_dict = torch.load(path_ + f'/{name}', map_location=lambda storage, loc:storage)
        self.dqn.load_state_dict(state_dict)

    def _plot(self, frame_idx, scores, losses, epsilons):
        clear_output(True)
        plt.figure(figsize=(10, 24))
        plt.subplot(311)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(312)
        plt.title('frame %s. loss: %s' % (frame_idx, np.mean(losses)))
        plt.plot(losses)
        plt.subplot(313)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()



class NN(nn.Module):
    def __init__(self, in_size, out_size, hid_size=128):
        super(NN, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_size, hid_size)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hid_size, hid_size)),
            ('relu2', nn.ReLU()),
            (nn.Linear(hid_size, out_size))
        ]))

    def forward(self, input):
        out = self.layers(input)
        return out


class DuelNN(nn.Module):
    def __init__(self, in_size, out_size, hid_size=128):
        super(DuelNN, self).__init__()

        self.feature = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_size, hid_size)),
            ('relu1', nn.ReLU())
        ]))

        self.advantage = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hid_size, hid_size)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hid_size, out_size)),
        ]))

        self.value = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hid_size, hid_size)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hid_size, 1)),
        ]))

    def forward(self, input):
        feature = self.feature(input)
        advantage = self.advantage(feature)
        value = self.value(feature)
        Qvalue = advantage + value - advantage.mean(dim=-1, keepdim=True)
        return Qvalue


class MemoryBuffer:
    def __init__(self, obs_size, buf_size, batch_size):
        self.max_size = buf_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.obs_buf = np.zeros([buf_size, obs_size], dtype=np.float32)
        self.next_obs_buf = np.zeros([buf_size, obs_size], dtype=np.float32)
        self.act_buf = np.zeros([buf_size], dtype=np.float32)
        self.rew_buf = np.zeros([buf_size], dtype=np.float32)
        self.done_buf = np.zeros([buf_size], dtype=np.float32)
        self.act_buf = np.zeros([buf_size], dtype=np.float32)

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.max_size, self.size + 1)

    def sample(self, beta=0.4):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.act_buf[idxs],
                    rews=self.rew_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(MemoryBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """
    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            alpha: float = 0.6
    ):
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


if __name__ == '__main__':
    env_id = 'CartPole-v0' # 'MountingCar-v0'
    env = gym.make(env_id)
    seed_all(env)
    # parameters
    num_frames = 200000
    memory_size = 1000
    batch_size = 32
    target_update = 100
    epsilon_decay = 1 / 100

    agent = D3QNAgent(env, memory_size, batch_size, target_update, epsilon_decay, conf=None)
    agent.train(num_frames, render_interval=1000)
