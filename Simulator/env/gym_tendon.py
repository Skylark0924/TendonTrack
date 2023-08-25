import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import os
import numpy as np
import json
import math
import shutil
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
import cv2
from scipy.misc.pilutil import imshow

from env.mujoco_model.continuum_env import *

HORIZON = 1000
GLOBAL_IMAGE_LOG_DIR = '/home/lab/Github/TendonTrack/Simulator/utils/global_image_log/'
VIDEO_LOG_DIR = '/home/lab/Github/TendonTrack/Simulator/utils/video_log/'


class TendonGymEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self, config=None):
        self.config = config
        self.env = create_env('Goal_LfD', None)
        self.reward = 0
        self.observation_space = Box(-1000 * np.ones(2), 1000 * np.ones(2))
        self.action_space = Box(-1 * np.ones(4), 1 * np.ones(4))
        self.prev_state = None

        self.image_num = 0
        self.timestep = 0
        self.reset_count = 0

        self.dis_rew_factor = 1  # TODO: Need to tune

        shutil.rmtree(GLOBAL_IMAGE_LOG_DIR)
        os.mkdir(GLOBAL_IMAGE_LOG_DIR)
        shutil.rmtree(VIDEO_LOG_DIR)
        os.mkdir(VIDEO_LOG_DIR)

        self.reset()

    def step(self, action):
        cur_state, cur_images = self.env.step(action)
        reward = self.get_reward(cur_state)

        global_fig = self.env.sim.render(width=960, height=720, camera_name='global_camera')
        cv2.imwrite(GLOBAL_IMAGE_LOG_DIR + "{}.jpg".format(self.image_num), global_fig)
        self.video.write(cv2.cvtColor(np.concatenate((cur_images, global_fig), axis=1), cv2.COLOR_BGR2RGB))

        self.timestep += 1
        self.image_num += 1
        # print('timestep: {}, current state: {}, reward: {}'.format(self.timestep, cur_state, reward))
        # --------------------done judgement------------------
        if cur_state[0] is None or cur_state[1] is None:
            cur_state = [500, 500]
            done = True
            self.video.release()
        elif abs(cur_state[0]) >= 500 or abs(cur_state[1]) >= 500:
            cur_state = [500, 500]
            reward = 100
            done = True
            self.video.release()
        elif abs(cur_state[0]) <= 5 and abs(cur_state[1]) <= 5:
            done = True
            self.video.release()
        elif self.timestep >= HORIZON:
            done = True
            self.video.release()
        else:
            done = False
        # ------------------------------------------------------
        return cur_state, reward, done, {}  # return next_state and reward

    def get_reward(self, cur_state):
        # reward function
        if cur_state[0] is None:
            reward = -100
        else:
            reward = (abs(self.prev_state[0]) - abs(cur_state[0])) + (abs(self.prev_state[1]) - abs(cur_state[1])) \
                        + self.dis_rew_factor * (abs(self.prev_state[2]) - abs(cur_state[2]))
        self.prev_state = cur_state
        # print('Reward: {}'.format(reward))
        return reward

    def reset(self):
        state, images = self.env.reset(0)
        # print('reset count: {}, timestep this episode: {}'.format(self.reset_count, self.timestep))

        self.video = cv2.VideoWriter(VIDEO_LOG_DIR + "sample{}.avi".format(self.reset_count),
                                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10.0, (1920, 720), True)
        self.timestep = 0
        self.reset_count += 1
        self.prev_state = state
        return state

    # def render(self, mode='human', close=False):
