import sys
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages')   
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages/ryu/app')   

import numpy as np
from gym import Env, spaces
from time import sleep

import proactive_mininet_api

NUMBER_HOSTS = 13
NUMBER_PATHS = 5
REWARD_SCALE = NUMBER_HOSTS*NUMBER_HOSTS*NUMBER_PATHS


class MininetEnv(Env):
    def __init__(self):
        self.number_of_requests = 0
        self.max_requests = 32
        self.done = False
        
        self.mininet_engine = proactive_mininet_api.MininetAPI(NUMBER_HOSTS, NUMBER_PATHS)
        
        self.observation_space = spaces.Box( \
            low=np.zeros((NUMBER_HOSTS,NUMBER_HOSTS,NUMBER_PATHS,1), dtype=np.float32), \
            high=np.full((NUMBER_HOSTS,NUMBER_HOSTS,NUMBER_PATHS,1), 100, dtype=np.float32), dtype=np.float32)
        
        self.action_space = spaces.Discrete(NUMBER_PATHS)
        self.state = self.mininet_engine.build_state()

    
    def step(self, action):
        self.mininet_engine.start_iperf(action)
        self.number_of_requests += 1
        reward = 0
        
        sleep(5)
                
        self.state = self.mininet_engine.build_state()
        for src in range(NUMBER_HOSTS):
            for dst in range(NUMBER_HOSTS):
                for path_number in range(NUMBER_PATHS):
                    bw = self.state[src, dst, path_number]
                    link = self.mininet_engine.get_state_helper().get(str(src + 1) + "_" + str(dst + 1) + "_" + str(path_number))
                    if link:
                        ex_link = link.split("_")
                        bw_percentage = self.mininet_engine.get_percentage(ex_link[0], ex_link[1], bw[0])
                        if bw_percentage is not None:
                            if bw_percentage > 75:
                                reward += 50
                            elif bw_percentage > 50:
                                reward += 30
                            elif bw_percentage > 25:
                                pass
                            elif bw_percentage > 0:
                                reward -= 10
                            else:
                                reward -= 100

                        
        if self.number_of_requests == self.max_requests:
            sleep(181)
            self.done = True
            
        return self.state, reward/REWARD_SCALE, self.done, {}
    
    def render():
        pass
    
    def get_state(self):
        return self.state
    
    def reset(self):  
        self.done = False
        self.mininet_engine.reset_measures()
        self.state = self.mininet_engine.build_state()
        self.number_of_requests = 0

        return self.state
