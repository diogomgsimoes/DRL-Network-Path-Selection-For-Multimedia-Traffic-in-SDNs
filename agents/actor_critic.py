import sys
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages') 

import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp
from matplotlib import pylab as plt

rewards_for_plot = []

def worker(t, worker_model, counter, params):
    worker_env = gym.make("DRL_Mininet-v0")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        print("Epoch:", i)
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env,worker_model)
        worker_env.reset()
        actor_loss,critic_loss,eplen = update_params(worker_opt,values,logprobs,rewards)
        counter.value = counter.value + 1
        
def run_episode(worker_env, worker_model):
    global rewards_for_plot
    state = torch.flatten(torch.from_numpy(worker_env.env.state.astype(np.float32))).reshape(1,320)
    values, logprobs, rewards = [],[],[]
    done = False
    j=0
    while (done == False):
        j+=1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, reward, done, info = worker_env.step(action.detach().numpy())
        state = torch.flatten(torch.from_numpy(state_.astype(np.float32))).reshape(1,320)
        rewards.append(reward)
    rewards_for_plot.append(sum(rewards))
    print("Reward:", sum(rewards))
    return values, logprobs, rewards

def update_params(worker_opt,values,logprobs,rewards,clc=0.1,gamma=0.95):
        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        Returns = []
        ret_ = torch.Tensor([0])
        for r in range(rewards.shape[0]):
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns,dim=0)
        actor_loss = -1*logprobs * (Returns - values.detach())
        critic_loss = torch.pow(values - Returns,2)
        loss = actor_loss.sum() + clc*critic_loss.sum()
        loss.backward()
        worker_opt.step()
        return actor_loss, critic_loss, len(rewards)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(320,160)
        self.l2 = nn.Linear(160,80)
        self.actor_lin1 = nn.Linear(80,5)
        self.l3 = nn.Linear(80,160)
        self.critic_lin1 = nn.Linear(160,1)
    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y),dim=0)
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic
    
MasterNode = ActorCritic()
MasterNode.share_memory()
processes = []
params = {
    'epochs':250,
    'n_workers':1,
}
counter = mp.Value('i',0)
for i in range(params['n_workers']):
    p = mp.Process(target=worker, args=(i,MasterNode,counter,params))
    p.start() 
    processes.append(p)
for p in processes:
    p.join()
for p in processes:
    p.terminate()
    
print(counter.value,processes[1].exitcode)

print('Plotting rewards ...')     
plt.figure(figsize=(10,7))
plt.plot(rewards_for_plot)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Return",fontsize=22)
plt.savefig('avg_return.png')
