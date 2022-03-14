import json
import os
import yaml
from argparse import Namespace, ArgumentParser
from utils import render_callback, fill_path
from rl_env import F110Env_Discrete
from dqn_agent import D3QNAgent
from ppo_agent import PPOAgent


def main(args):
    task = args.task
    step = args.load_step
    print('Initialize RL Environment')
    with open('../config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    env = F110Env_Discrete(conf=conf)
    env.f110.add_render_callback(render_callback)

    cfg_path = os.path.join(fill_path('config'), f'rlf110_{task}cfg.json')
    cfg = Namespace(**json.load(open(cfg_path)))
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    #print(cfg)

    if task == 'ddqn':
        agent = D3QNAgent(env, cfg.memory_size, cfg.batch_size, cfg.target_update, cfg.epsilon_decay, conf=cfg)

    else:
        agent = PPOAgent(env=env, lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, gamma=cfg.gamma, K_epoch=cfg.K_epoch,
                         eps_clip=cfg.eps_clip, conf=cfg)
    if step is not None:
        agent.load(f'step_{step}')
    agent.train(1000000, update_interval=100, render_interval=1000000, save_interval=2000, eval_episode=1000000, initial_ep=step)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, help='Enter the agent name', default='ddqn')
    parser.add_argument('--load_step', type=int, help='Start from loaded model', default=None)

    args = parser.parse_args()
    main(args)