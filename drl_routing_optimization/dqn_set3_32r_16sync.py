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

        self.requests_bw = [5, 10, 15, 18, 20]

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

    def build_state(self):
        state = np.empty((NUMBER_OF_HOSTS, NUMBER_OF_HOSTS, NUMBER_OF_PATHS, 1), dtype=object)

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


if __name__ == "__main__":
    engine = DRLEngine()
    env = RoutingEnv()

    n_state = 845
    n_action = 5

    l1 = 845
    l2 = 1500
    l3 = 700
    l4 = 200
    l5 = 5

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4),
        torch.nn.ReLU(),
        torch.nn.Linear(l4, l5)
    )

    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())

    gamma = 0.9
    epsilon = 0.5
    learning_rate = 1e-3

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    state_flattened_size = 845
    losses = []
    total_reward_list = []
    epochs = 5000
    mem_size = 50000
    batch_size = 256
    sync_freq = 16
    replay = deque(maxlen=mem_size)

    for i in range(epochs):
        print("Starting training, epoch:", i)
        cnt = 0
        total_reward = 0
        _state = env.get_state()
        state1 = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, state_flattened_size)
        done = False
        env.reset()

        while not done:
            print("Step:", cnt + 1)
            cnt += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            if (random.random() < epsilon):
                action_ = np.random.randint(0, n_action - 1)
            else:
                action_ = np.argmax(qval_)

            state, reward, done, _ = env.step(action_)
            state2 = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, state_flattened_size)

            exp = (state1, action_, reward, state2, done)

            replay.append(exp)
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model2(state2_batch)

                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
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

        if epsilon > 0.01:
            epsilon -= (1 / epochs)

    print(total_reward_list)
    torch.save(model.state_dict(), 'dqn_policy_set3_32r_16sync.pt')

    sizes = [25, 50, 100, 200, 500]
    for size in sizes:
        avg = []
        for idx in range(0, len(total_reward_list), size):
            avg += [sum(val for val in total_reward_list[idx:idx + size]) / size]

        plt.figure(figsize=(10, 7))
        plt.plot(avg)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Return", fontsize=22)
        plt.savefig('dqn_policy_set3_32r_16sync_{}.png'.format(size))
