import sys

sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages')   

from collections import defaultdict

import numpy as np
from gym import Env, spaces
import json
from random import choice
from time import sleep

import mininet_api


class MininetEnv(Env):
    def __init__(self):
        self.number_of_paths = 5
        # self.number_of_requests = 0
        self.max_requests = 3
        self.done = False
        
        self.mininet_engine = mininet_api.MininetAPI(5, 60, 100)
        
        self.observation_space = spaces.Box(np.array([0]), np.array([30]))
        self.action_space = spaces.Discrete(self.number_of_paths)
        #self.mininet_engine.build_action_space(self.number_of_paths)
    
    def step(self, action):
        # action: start iperf between two hosts
        # while self.number_of_requests != self.max_requests:
            # numbers = list(range(1, self.mininet_engine.number_hosts+1))
            # src_id = choice(numbers)
            # numbers.remove(src_id)
            
            # action: start iperf between two hosts
            # "h{}".format(src_id), "h{}".format(choice(numbers))
        self.mininet_engine.start_iperf()
            
            # self.number_of_requests += 1
            
        sleep(40)
        
        # state: read iperf reports
        n_state = self.mininet_engine.build_state()
        
        reward = 0
        
        # reward: evaluate state
        for src in range(1, 10):
            for dst in range(1, 10):
                if n_state[src][dst] != None:
                    avg_metric = sum(n_state[src][dst])/len(n_state[src][dst])
                    if avg_metric > 22:
                        reward += 10
                    elif avg_metric > 15: 
                        reward += 5
                    elif avg_metric > 7: 
                        reward -= 5
                    else: 
                        reward -= 10
        
        self.done = True
        info = {}
        
        return n_state, reward, self.done, info
    
    def render():
        pass
    
    def reset(self):
        self.done = False
        # self.number_of_requests = 0
        self.mininet_engine.reset_measures()
