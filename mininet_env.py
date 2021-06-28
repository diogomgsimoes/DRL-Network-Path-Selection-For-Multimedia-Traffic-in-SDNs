#!/usr/bin/python

import time
import gym
import numpy as np
from mininet_api import MininetAPI


class MininetEnv(gym.Env):
    def __init__(self):
        self.LINK_BW = 100
        self.NUMBER_NODES = 10
        self.MAX_TICKS = 1000

        self.mininet_engine = MininetAPI(self.LINK_BW)

        self.replay_buffer = []
        self.episode_over = False
        self.curr_episode = -1

        # Transfer, Bitrate, Jitter, Packet loss
        self.observation_space = np.full((self.NUMBER_NODES, self.NUMBER_NODES), [0.0, 0.0, 0.0, 0.0])
        self.action_space = np.full((self.NUMBER_NODES, self.NUMBER_NODES), 1)

    def step(self):
        state = self.get_state()
        action = self.take_action()
        reward = self.get_reward()
        time.sleep(2)
        transformed_state = self.get_state()
        return state, reward, action, transformed_state

    def take_action(self):
        pass

    def get_reward(self):
        measures = self.mininet_engine.get_measures()
        balanced_reward = 0
        balanced_reward += float(measures[0])/self.LINK_BW
        # continue with the remaining metrics
        return balanced_reward

    def get_state(self):
        pass

    def reset(self):
        pass


if __name__ == '__main__':
    print("Environment started!")
    print("Measuring...")
    mininet_env = MininetEnv()
    print(mininet_env.step())
