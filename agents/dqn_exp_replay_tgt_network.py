import sys
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages') 

import numpy as np
import torch
import gym
import random
from matplotlib import pylab as plt
from collections import deque
import copy
import time
import itertools

start = time.time()

l1 = 320
l2 = 160
l3 = 80
l4 = 5

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
)

model2 = copy.deepcopy(model)
model2.load_state_dict(model.state_dict())

gamma = 0.9
epsilon = 0.5
learning_rate = 1e-3

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
total_reward_list = []
epochs = 500
mem_size = 64
batch_size = 8
sync_freq = 24
replay = deque(maxlen=mem_size)

env = gym.make('DRL_Mininet-v0')
print("Env started.")

for i in range(epochs):
    print("Starting training, epoch:", i)
    cnt = 0
    total_reward = 0
    _state = env.get_state()
    state1 = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1,320)
    done = False
    env.reset()
    
    while not done: 
        print("Step:", cnt+1)
        cnt += 1
        qval = model(state1) 
        qval_ = qval.data.numpy()
        if (random.random() < epsilon):
            action_ = np.random.randint(0,4)
        else:
            action_ = np.argmax(qval_)
        
        state, reward, done, _ = env.step(action_)
        state2 = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1,320)
        
        exp = (state1, action_, reward, state2, done)
        replay.append(exp)
        state1 = state2
        
        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
            Q1 = model(state1_batch) 
            with torch.no_grad():
                Q2 = model2(state2_batch)
            
            Y = reward_batch + gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            print(i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            if cnt % sync_freq == 0:
                model2.load_state_dict(model.state_dict())
        
        total_reward += reward
    
    total_reward_list.append(total_reward)
    print("Episode reward:", total_reward)
        
    if epsilon > 0.1:
        epsilon -= (1/epochs)
        
print(total_reward_list)
   
print('Plotting losses ...')     
plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Loss",fontsize=22)
plt.savefig('avg_loss.png') 

print('Plotting rewards ...')     
plt.figure(figsize=(10,7))
plt.plot(total_reward_list)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Return",fontsize=22)
plt.savefig('avg_return.png')

torch.save(model.state_dict(), 'dqn_model_exp_replay_target_network.pt')

curr_time = time.time()
print("Running time:", time.strftime('%H:%M:%S', time.gmtime(curr_time-start)))
