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
from utils.teacher import PurePursuitExpert


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

    obs = env.reset()
    i=0
    while True:
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

        action = model(obs)
        action = action.squeeze().data.cpu().numpy()
        
        obs, reward, done, info = env.step(action)
        prev_screen = env.render(mode='rgb_array')
        plt.imshow(prev_screen)
        plt.savefig(f"figs/Fig{i}.png")
        i+=1
        ipythondisplay.clear_output(wait=True)
        ipythondisplay.display(plt.gcf())
        if done:
            if reward < 0:
                print('*** FAILED ***')
                time.sleep(0.7)
                
            obs = env.reset()
            env.render(mode='rgb_array')

def colab_enjoy():
  _enjoy()
  
if __name__ == '__main__':
    _enjoy()