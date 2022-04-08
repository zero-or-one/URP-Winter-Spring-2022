import random

import matplotlib.pyplot as plt
from f110_gym.envs.f110_env import F110Env
from gym import spaces
import yaml
import numpy as np
from argparse import Namespace
from math import pi
from utils import render_callback

class F110Env_Discrete:
    def __init__(self, speed=3, conf=None, num_agents=1):
        self.f110 = F110Env(map=conf.map_path, map_ext=conf.map_ext, num_agents=num_agents)
        self.conf = conf
        self.speed = speed
        self.action_space = spaces.Discrete(3) # for now 3 actions
        self.delta = 0.1
        self.num_agents = num_agents

        self.action = np.array([
            [1, self.speed],
            [-1, self.speed],
            [0, self.speed],
            #[1, self.speed+self.delta],
            #[-1, self.speed+self.delta],
            #[0, self.speed+self.delta],
            #[1, self.speed-self.delta],
            #[-1, self.speed-self.delta],
            #[0, self.speed-self.delta],
        ])
        self.observation_space = spaces.Box(low=0, high=1000, shape=(27,1))


        with open(conf.wpt_path, encoding='utf-8') as f:
            self.waypoints = np.loadtxt(f, delimiter=';')
            self.waypoints_param = np.vstack([self.waypoints[:,1], self.waypoints[:,2], self.waypoints[:,3] + pi/2]).T

    def get_action(self, idx: int) -> np.ndarray:
        if self.num_agents < 2:
            action = self.action[idx].reshape(1, -1)
        else:
            action = np.array([self.action[i].reshape(1, -1)[0] for i in idx])
            #print(action)
        return action

    def get_obs(self, raw_obs: dict) -> np.ndarray:
        obs = raw_obs['scans'][0][::40]
        return obs

    def reset(self):
        poses = []
        if self.num_agents < 2:
            idx = random.sample(range(len(self.waypoints_param)), 1)
            pos = self.waypoints_param[idx]
            params = self.f110.reset(pos)
            obs = self.get_obs(params[0])
        else:
            for i in range(self.num_agents):
                idx = random.sample(range(len(self.waypoints_param)), 1)
                pos = self.waypoints_param[idx]
                poses.append(pos[0])
            params = self.f110.reset(np.array(poses))
            obs = self.get_obs(params[0])
        return obs

    def step(self, action):
        action = self.get_action(action)
        count = 3
        done = False
        while count > 0 and not done:
            raw_obs, reward, done, info = self.f110.step(action)
            count -= 1
        if done:
            reward -= 0.5
        obs = self.get_obs(raw_obs)
        #print(raw_obs)
        if min(obs) < 0.4:
            reward -= 0.02
        reward *= 2
        return obs, reward, done, info

    def render(self, mode='human'):
        self.f110.render(mode)

    def close(self):
        self.f110.close()

if __name__ == '__main__':
    with open('./config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = F110Env_Discrete(conf=conf, num_agents=3)
    env.f110.add_render_callback(render_callback)

    for i_ep in range(30):
        obs = env.reset()
        done = False
        i = 0
        min_obs = []
        env.render()
        while not done:
            i+=1
            env.close()
            env.render()
            action_ids = [random.randint(0, 2) for i in range(env.num_agents)]
            obs, reward, done, info = env.step(action_ids)
            print('obs', obs)
            min_obs.append(min(obs))
            print('reward', reward)
            #if i % 30 == 0:
            #    plt.plot(obs)
            #    plt.tittle('obs')
            #    plt.show()
            print("Finish episode")
            #plt.plot(min_obs)
            #plt.title(min_obs)
            #plt.show()

