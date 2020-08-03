#!/usr/bin/env python3

"""
Control the simulator or Duckiebot using a model trained with imitation
learning, and visualize the result.
"""

import time
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.append('/content/gym-duckietown/src')

import argparse
import math
from IPython import display as ipythondisplay
import torch

import numpy as np
import gym
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay

from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from gym.wrappers import Monitor
from utils.teacher import PurePursuitExpert
from utils.utils import compute_dist

from imitation.pytorch.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _enjoy():
    model = Model(action_dim=2, max_action=1.)
    
    try:
        state_dict = torch.load('models/imitate.pt', map_location=device)
        model.load_state_dict(state_dict)
    except:
        print('failed to load model')
        exit()

    model.eval().to(device)

    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    env = Monitor(env, directory='model_video', force=True)

    obs = env.reset()
    dists = []
    while True:
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

        action = model(obs)
        action = action.squeeze().data.cpu().numpy()
        obs, reward, done, info = env.step(action)
        dists.append(compute_dist(env))
        prev_screen = env.render(mode='rgb_array')
        if done:
            if reward < 0:
                print('*** FAILED ***')
                time.sleep(0.7)
                
            obs = env.reset()
            env.render(mode='rgb_array')
            with open('mean_distance.txt', 'a') as f:
                string = f"Time: {time.time()} | Mean distance: {np.mean(dists)}\n"
                f.write(string)

def colab_enjoy():
  _enjoy()
  
if __name__ == '__main__':
    _enjoy()