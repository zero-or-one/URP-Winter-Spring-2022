import json
import os
import yaml
from argparse import Namespace, ArgumentParser
from utils import render_callback, fill_path
from rl_env import F110Env_Discrete
from dqn_agent import D3QNAgent
from ppo_agent import PPOAgent
import numpy as np

def main(args):
    task = args.task
    model_name = args.model
    print('Initialize RL Environment')
    with open('./config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    env = F110Env_Discrete(conf=conf)
    env.f110.add_render_callback(render_callback)

    cfg_path = os.path.join(fill_path('config'), f'RlF110_{task}cfg.json')
    cfg = Namespace(**json.load(open(cfg_path)))
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    if task == 'ddqn':
        agent = D3QNAgent(env, cfg.memory_size, cfg.batch_size, cfg.target_update, cfg.epsilon_decay, conf=cfg)
    else:
        agent = PPOAgent(env=env, lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, gamma=cfg.gamma, K_epoch=cfg.K_epoch,
                         eps_clip=cfg.eps_clip, conf=cfg)
    score = agent.eval(reload_model=True, render=True, model=model_name)
    print("Agent's average score is:", np.round(score, 4))
    #agent.test(render_times=5, render=True)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, help='Enter the agent name', default='ddqn')
    parser.add_argument('--model', type=str, help='Enter the trained model name', default=None)

    args = parser.parse_args()
    main(args)