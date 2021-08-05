import sys
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages')   
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages/ryu/app')   

import numpy as np
from gym import Env, spaces
from time import sleep

import proactive_mininet_api

NUMBER_HOSTS = 8
NUMBER_PATHS = 5
REWARD_SCALE = NUMBER_HOSTS*NUMBER_HOSTS*NUMBER_PATHS


class MininetEnv(Env):
    def __init__(self):
        self.number_of_requests = 0
        self.max_requests = 3
        self.done = False
        
        self.mininet_engine = proactive_mininet_api.MininetAPI(NUMBER_HOSTS, NUMBER_PATHS)
        
        self.observation_space = spaces.Box( \
            low=np.zeros((NUMBER_HOSTS,NUMBER_HOSTS,NUMBER_PATHS,1), dtype=np.float32), \
            high=np.full((NUMBER_HOSTS,NUMBER_HOSTS,NUMBER_PATHS,1), 102400, dtype=np.float32), dtype=np.float32)
        
        self.state = np.zeros((NUMBER_HOSTS,NUMBER_HOSTS,NUMBER_PATHS,1), dtype=np.float32)
        self.action_space = spaces.Discrete(NUMBER_PATHS)
    
    def step(self, action):
        # action: start iperf between two hosts
        self.mininet_engine.start_iperf(action, self.number_of_requests)
        self.number_of_requests += 1
        
        print("ACTION:", action)
        print("REQUEST NUMBER:", self.number_of_requests)
            
        reward = 0
        self.done = False
        info = {}
        
        sleep(5)
                
        # state: read iperf reports
        self.state = self.mininet_engine.build_state()
    
        # reward: evaluate state
        for src in range(NUMBER_HOSTS):
            for dst in range(NUMBER_HOSTS):
                for path_number in range(NUMBER_PATHS):
                    metric = self.state[src, dst, path_number]
                    if metric != None:
                        metric_percentage = (metric/102400)*100
                        if metric_percentage > 80:
                            reward += 10
                        elif metric_percentage > 50: 
                            reward += 5
                        elif metric_percentage > 30: 
                            reward -= 5
                        else: 
                            reward -= 10
                            
        print("REWARD:", reward/REWARD_SCALE)
                        
        if self.number_of_requests == self.max_requests:
            sleep(25)
            self.done = True
            
        return self.state, reward/REWARD_SCALE, self.done, info
    
    def render():
        pass
    
    def reset(self):
        self.done = False
        self.state = np.zeros((NUMBER_HOSTS,NUMBER_HOSTS,NUMBER_PATHS,1), dtype=np.float32)
        self.number_of_requests = 0
        self.mininet_engine.reset_measures()
        return self.state
