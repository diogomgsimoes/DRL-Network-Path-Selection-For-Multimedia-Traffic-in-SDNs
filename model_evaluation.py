import sys
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages/gym/envs/classic_control')     

import mininet_env
import torch
from torch import nn
import numpy as np
from torch.nn.modules.activation import LeakyReLU

l1 = 845
l2 = 1500
l3 = 700
l4 = 200
l5 = 5

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(845, 1500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1500, 700)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(700, 200)
        self.fc_adv = nn.Linear(700, 200)

        self.value = nn.Linear(200, 1)
        self.adv = nn.Linear(200, 5)

    def forward(self, state):
        y = self.relu(self.fc1(state))
        y = self.relu(self.fc2(y))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()

model_test = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4),
    torch.nn.ReLU(),
    torch.nn.Linear(l4, l5)
)

env = mininet_env.MininetEnv()

def test_model(model):
    _state = env.reset()
    state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, l1)
    done = False
    rewards = []
    while not done:
        qval = model(state)
        qval_ = qval.data.numpy()
        action = np.argmax(qval_)
        print("State:", state)
        # action = model_test.select_action(state)
        print("Path:", action)
        _state, reward, done, _ = env.step(action)
        state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, l1)
        # print(state)
        rewards.append(reward)
        print("Request:", len(rewards), " Reward:", reward)
    print("Reward sum:", sum(rewards))
    
if __name__ == "__main__":
    # model_test = QNetwork()
    model_test.load_state_dict(torch.load("dqn_policy_set4.pt"))
    # model_test.load_state_dict(torch.load("dueling_dqn-policy.pt"))
    test_model(model_test)
