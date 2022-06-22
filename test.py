import gym
import pygame
from gym.utils.play import play
import random
import numpy as np
import gym
import yaml
from gym_qRacing.envs.functions import Helper

with open('racesim_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
play(gym.make("qRacing-base-v0", config=config), keys_to_action=mapping)