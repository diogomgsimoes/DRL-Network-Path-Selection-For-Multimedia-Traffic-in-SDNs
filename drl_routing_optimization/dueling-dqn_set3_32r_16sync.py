import torch
import copy
import random

import networkx as nx
import numpy as np

from itertools import islice
from gym import Env
from gym.spaces import Discrete, Box
from matplotlib import pylab as plt
from collections import deque
from torch.nn import functional as F
from torch import nn

TOPOLOGY_FILE_NAME = 'topology_ARPANET.txt'
NUMBER_OF_HOSTS = 13
NUMBER_OF_PATHS = 5
REWARD_SCALE = NUMBER_OF_HOSTS * NUMBER_OF_HOSTS * NUMBER_OF_PATHS


class DRLEngine():
    def __init__(self):
        self.graph = nx.Graph()
        self.link_bw_capacity = {}
        self.current_link_bw = {}
        self.hosts = {}
        self.paths = {}

        self.host_pairs = [('H4', 'H8'), ('H2', 'H11'), ('H2', 'H13'), ('H2', 'H9'), ('H4', 'H11'), ('H4', 'H9'), ('H2', 'H8'), ('H1', 'H11'),
                        ('H1', 'H9'), ('H4', 'H13'), ('H4', 'H10'), ('H4', 'H7'), ('H3', 'H8'), ('H2', 'H10'), ('H2', 'H7'), ('H1', 'H8'),
                        ('H4', 'H12'), ('H3', 'H11'), ('H2', 'H12'), ('H1', 'H13'), ('H3', 'H9'), ('H1', 'H12'), ('H1', 'H7'), ('H4', 'H6'),
                        ('H3', 'H10'), ('H5', 'H6'), ('H3', 'H13'), ('H3', 'H7'), ('H7', 'H6'), ('H5', 'H11'), ('H5', 'H8'), ('H3', 'H12')]

        self.requests_bw = [3, 5, 10, 15, 18]

        self.upload_topology()
        self.build_graph()
        self.calculate_paths()

    def upload_topology(self):
        with open(TOPOLOGY_FILE_NAME, 'r') as topo:
            for row in topo.readlines():
                row_data = row.split()
                if 'H' in row_data[0]:
                    self.hosts[row_data[0]] = row_data[1].replace("S", "")
                elif 'S' in row_data[0]:
                    src_id = row_data[0].replace("S", "")
                    dst_id = row_data[1].replace("S", "")
                    self.link_bw_capacity[(src_id, dst_id)] = int(row_data[2])
                    self.link_bw_capacity[(dst_id, src_id)] = int(row_data[2])

        self.current_link_bw = copy.deepcopy(self.link_bw_capacity)

    def build_graph(self):
        with open(TOPOLOGY_FILE_NAME, 'r') as topo:
            for line in topo.readlines():
                nodes = line.split()[:2]
                for node in nodes:
                    if not self.graph.has_node(node):
                        self.graph.add_node(node)
                self.graph.add_edge(nodes[0], nodes[1])

    def k_shortest_paths(self, graph, source, target, k):
        try:
            calc = list(islice(nx.shortest_simple_paths(graph, source, target), k))
        except nx.NetworkXNoPath:
            calc = []

        return [path for path in calc]

    def calculate_paths(self):
        for src_host_id in range(1, NUMBER_OF_HOSTS + 1):
            src = "H{}".format(src_host_id)
            for dst_host_id in range(1, NUMBER_OF_HOSTS + 1):
                dst = "H{}".format(dst_host_id)
                if self.graph.has_node(src) and self.graph.has_node(dst):
                    self.paths[(src, dst)] = self.k_shortest_paths(self.graph, src, dst, NUMBER_OF_PATHS)
                    for path in self.paths[(src, dst)]:
                        if len(path) != 0:
                            for i in range(0, len(path)):
                                if "S" in path[i]:
                                    path[i] = path[i].replace("S", "")
                                    path[i] = int(path[i])

    def make_reservation(self, path_id):
        pair = self.host_pairs.pop(0)
        path = self.paths[(pair[0], pair[1])][path_id]
        request_bw = self.requests_bw[random.randint(0, 4)]

        for s1, s2 in zip(path[:-1], path[1:]):
            if self.current_link_bw.get((str(s1), str(s2))):
                self.current_link_bw[(str(s1), str(s2))] -= request_bw
                if self.current_link_bw[(str(s1), str(s2))] == 0:
                    self.current_link_bw[(str(s1), str(s2))] = 1
            if self.current_link_bw.get((str(s2), str(s1))):
                self.current_link_bw[(str(s2), str(s1))] -= request_bw
                if self.current_link_bw[(str(s2), str(s1))] == 0:
                    self.current_link_bw[(str(s2), str(s1))] = 1

    def get_percentage(self, src, dst, bw):
        if self.link_bw_capacity.get((src, dst)):
            return (bw / self.link_bw_capacity.get((src, dst))) * 100
        else:
            return None

    def get_state_helper(self):
        return self.state_helper

    def build_state(self):
        state = np.empty((NUMBER_OF_HOSTS, NUMBER_OF_HOSTS, NUMBER_OF_PATHS, 1), dtype=object)

        for src in range(1, NUMBER_OF_HOSTS + 1):
            h_src = "H{}".format(src)
            for dst in range(1, NUMBER_OF_HOSTS + 1):
                h_dst = "H{}".format(dst)
                cnt = 0
                if len(self.paths[(h_src, h_dst)]) == 1:
                    if not self.paths[(h_src, h_dst)]:
                        for idx in range(NUMBER_OF_PATHS):
                            state[src - 1, dst - 1, idx] = -1
                    else:
                        state[src - 1, dst - 1, 0] = 100
                        for idx in range(1, NUMBER_OF_PATHS):
                            state[src - 1, dst - 1, idx] = -1
                else:
                    for path in self.paths[(h_src, h_dst)]:
                        min_value = float('Inf')
                        for s1, s2 in zip(path[:-1], path[1:]):
                            stats = self.current_link_bw.get((s1, s2))
                            if stats:
                                if float(stats) < float(min_value):
                                    min_value = self.current_link_bw[(s1, s2)]
                                    self.state_helper[str(src) + "_" + str(dst) + "_" + str(cnt)] = str(s1) + "_" + str(s2)

                        state[src - 1, dst - 1, cnt] = float(min_value)
                        cnt += 1

                    for idx in range(len(self.paths[(h_src, h_dst)]), NUMBER_OF_PATHS):
                        state[src - 1, dst - 1, idx] = -1
        return state

    def reset(self):
        self.graph = nx.Graph()

        self.host_pairs = [('H4', 'H8'), ('H2', 'H11'), ('H2', 'H13'), ('H2', 'H9'), ('H4', 'H11'), ('H4', 'H9'), ('H2', 'H8'), ('H1', 'H11'),
                      ('H1', 'H9'), ('H4', 'H13'), ('H4', 'H10'), ('H4', 'H7'), ('H3', 'H8'), ('H2', 'H10'), ('H2', 'H7'), ('H1', 'H8'),
                      ('H4', 'H12'), ('H3', 'H11'), ('H2', 'H12'), ('H1', 'H13'), ('H3', 'H9'), ('H1', 'H12'), ('H1', 'H7'), ('H4', 'H6'),
                      ('H3', 'H10'), ('H5', 'H6'), ('H3', 'H13'), ('H3', 'H7'), ('H7', 'H6'), ('H5', 'H11'), ('H5', 'H8'), ('H3', 'H12')]

        self.current_link_bw = copy.deepcopy(self.link_bw_capacity)


class RoutingEnv(Env):
    def __init__(self):
        self.requests = 0
        r = int(np.random.normal(24, 8))
        while r > 32 or r < 1:
            r = int(np.random.normal(24, 8))
        self.max_requests = r
        self.done = False

        self.engine = DRLEngine()

        self.observation_space = Box(
            low=np.zeros((NUMBER_OF_HOSTS, NUMBER_OF_HOSTS, NUMBER_OF_PATHS, 1), dtype=np.float32),
            high=np.full((NUMBER_OF_HOSTS, NUMBER_OF_HOSTS, NUMBER_OF_PATHS, 1), 100, dtype=np.float32),
            dtype=np.float32)

        self.action_space = Discrete(NUMBER_OF_PATHS)
        self.state = np.full((NUMBER_OF_HOSTS, NUMBER_OF_HOSTS, NUMBER_OF_PATHS, 1), 100, dtype=np.float32)

   def step(self, action):
        self.engine.make_reservation(action)
        self.requests += 1

        reward = 0
        self.state = self.engine.build_state()

        for src in range(NUMBER_OF_HOSTS):
            for dst in range(NUMBER_OF_HOSTS):
                for path_number in range(NUMBER_OF_PATHS):
                    bw = self.state[src, dst, path_number]
                    link = self.engine.get_state_helper().get(
                        str(src + 1) + "_" + str(dst + 1) + "_" + str(path_number))
                    if link:
                        ex_link = link.split("_")
                        bw_percentage = self.engine.get_percentage(ex_link[0], ex_link[1], bw[0])
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

        if self.requests == self.max_requests:
            self.done = True

        return self.state, (reward / REWARD_SCALE), self.done, {}
    
    def render(self):
        pass

    def get_state(self):
        return self.state

    def reset(self):
        self.done = False
        self.engine.reset()
        self.state = self.engine.build_state()
        self.requests = 0

        r = int(np.random.normal(24, 8))
        while r > 32 or r < 1:
            r = int(np.random.normal(24, 8))
        self.max_requests = r

        return self.state


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


env = RoutingEnv()
n_state = 845
n_action = 5

onlineQNetwork = QNetwork().to(device)
targetQNetwork = QNetwork().to(device)
targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)

GAMMA = 0.99
EXPLORE = 20000
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 256

UPDATE_STEPS = 16

memory_replay = Memory(REPLAY_MEMORY)

epsilon = INITIAL_EPSILON
learn_steps = 0
begin_learn = False

episode_reward = 0
total_rewards = []

for epoch in range(5000):
    state = env.reset()
    _state = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, n_state)
    episode_reward = 0
    for time_steps in range(200):
        p = random.random()
        if p < epsilon:
            action = random.randint(0, n_action-1)
        else:
            tensor_state = _state.to(device)
            action = onlineQNetwork.select_action(tensor_state)
        next_state, reward, done, _ = env.step(action)
        _next_state = torch.flatten(torch.from_numpy(next_state.astype(np.float32))).reshape(1, n_state)
        episode_reward += reward
        memory_replay.add((_state, _next_state, action, reward, done))
        if memory_replay.size() > 256:
            if begin_learn is False:
                print('learn begin!')
                begin_learn = True
            learn_steps += 1
            if learn_steps % UPDATE_STEPS == 0:
                targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
            batch = memory_replay.sample(BATCH, False)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

            batch_state = torch.cat([item for item in batch_state]).to(device)
            batch_next_state = torch.cat([item for item in batch_next_state]).to(device)
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

            with torch.no_grad():
                onlineQ_next = onlineQNetwork(batch_next_state)
                targetQ_next = targetQNetwork(batch_next_state)
                online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
                y = batch_reward + (1 - batch_done) * GAMMA * targetQ_next.gather(1, online_max_action.long())

            loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        if done:
            total_rewards.append(episode_reward)
            break
        _state = _next_state

    if epoch % 10 == 0:
        torch.save(onlineQNetwork.state_dict(), 'dueling-dqn_policy_set3_32r_16sync')
        print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))

print(total_rewards)

sizes = [25, 50, 100, 200, 500]
for size in sizes:
    avg = []
    for idx in range(0, len(total_rewards), size):
        avg += [sum(val for val in total_rewards[idx:idx + size]) / size]

    plt.figure(figsize=(10, 7))
    plt.plot(avg)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Return", fontsize=22)
    plt.savefig('dueling-dqn_policy_set3_32r_16sync_{}.png'.format(size))
