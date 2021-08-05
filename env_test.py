import sys
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages/gym/envs/classic_control')     

import mininet_env
from random import randint
import time

env = mininet_env.MininetEnv()
episodes = 10

time.sleep(10)

for episode in range(1, episodes+1):
    env.reset()
    done = False
    score = 0

    while not done:
        n_state, reward, done, info = env.step(randint(0, 4))
        score += reward

    print('Episode: {}, Score: {}'.format(episode, score))
