import json
import numpy as np
import os
import torch

def seed_all(env, seed=13):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determenistic = True
    np.random.seed(seed)
    env.seed(seed)

def fill_path(path):
    return os.path.abspath(os.path.join('.', path))

def fill_config(task):
    res_dir = fill_path('result')
    default_config = dict(
        task_name = task,
        res_dir = os.path.join(res_dir, task),
        log_dir = os.path.join(res_dir, task, 'logs'),
        ckpt_dir = os.path.join(res_dir, task, 'checkpoints')
    )
    return default_config

def get_config(task):
    cfg = fill_config(task)
    path_ = os.path.join(fill_path('config'))
    os.makedirs(path_, exists_ok=True)
    json.dump(cfg, open(os.path.join(path_, f'rl_single_{task}_config.json'), 'w'), indent=3)

def render_callback(env_renderer):
    # custom extra drawing function
    e = env_renderer
    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800
