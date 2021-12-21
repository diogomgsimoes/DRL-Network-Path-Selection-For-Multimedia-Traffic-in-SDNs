import torch
import copy
import collections
import random

import networkx as nx
import numpy as np

from itertools import islice
from gym import Env
from gym.spaces import Discrete, Box
from matplotlib import pylab as plt
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR


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

class QNetwork(nn.Module):
    def __init__(self, action_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(845, 1500)
        self.fc_2 = nn.Linear(1500, 700)
        self.fc_3 = nn.Linear(700, 200)
        self.fc_4 = nn.Linear(200, action_dim)

    def forward(self, inp):
        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = F.leaky_relu(self.fc_3(x1))
        x1 = self.fc_4(x1)

        return x1


class Memory(object):
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        n = len(self.is_done)
        idx = random.sample(range(0, n - 1), batch_size)

        return torch.Tensor(self.state)[idx].to(device), torch.LongTensor(self.action)[idx].to(device), \
               torch.Tensor(self.state)[1 + np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()


def select_action(model, env, state, eps):
    _state = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, 845)
    state = _state.to(device)

    with torch.no_grad():
        values = model(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = np.argmax(values.cpu().numpy())

    return action


def train(batch_size, current, target, optim, memory, gamma):
    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    _states = states.reshape(256, 845)
    _next_states = next_states.reshape(256, 845)

    q_values = current(_states)

    next_q_values = current(_next_states)
    next_q_state_values = target(_next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()


def evaluate(Qmodel, env, repeats):
    Qmodel.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            _state = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, 845).to(device)
            with torch.no_grad():
                values = Qmodel(_state)
            action = np.argmax(values.cpu().numpy())
            state, reward, done, _ = env.step(action)
            perform += reward
    Qmodel.train()
    return perform / repeats


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


if __name__ == "__main__":
    engine = DRLEngine()
    env = RoutingEnv()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_state = 845
    n_action = 5

    gamma=0.99
    lr=1e-3
    min_episodes=8
    eps=1
    eps_decay=0.995
    eps_min=0.01
    update_step=16
    batch_size=256
    update_repeats=16
    num_episodes=5000
    seed=42
    max_memory_size=500000
    lr_gamma=0.9
    lr_step=100
    measure_step=1
    measure_repeats=1

    total_rewards = []

    torch.manual_seed(seed)
    env.seed(seed)

    Q_1 = QNetwork(action_dim=env.action_space.n).to(device)
    Q_2 = QNetwork(action_dim=env.action_space.n).to(device)

    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_1, Q_2)

    # we only train Q_1
    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
    optimizer.step()
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []

    for episode in range(num_episodes):
        if episode % measure_step == 0:
            performance.append([episode, evaluate(Q_1, env, measure_repeats)])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            total_rewards.append(performance[-1][1])
            print("lr: ", scheduler.get_last_lr()[0])
            print("eps: ", eps)

        state = env.reset()
        memory.state.append(state.reshape(1, 845))

        done = False
        i = 0
        while not done:
            i += 1
            action = select_action(Q_2, env, state, eps)
            state, reward, done, _ = env.step(action)

            # save state, action, reward sequence
            memory.update(state.reshape(1, 845), action, reward, done)

        if episode >= min_episodes and episode % update_step == 0:
            for _ in range(update_repeats):
                train(batch_size, Q_1, Q_2, optimizer, memory, gamma)

            # transfer new parameter from Q_1 to Q_2
            update_parameters(Q_1, Q_2)

        # update learning rate and eps
        scheduler.step()
        eps = max(eps * eps_decay, eps_min)

    torch.save(Q_1.state_dict(), "ddqn_policy_set3_32r_16sync.pt")

    sizes = [25, 50, 100, 200, 500]
    for size in sizes:
        avg = []
        for idx in range(0, len(total_rewards), size):
            avg += [sum(val for val in total_rewards[idx:idx + size]) / size]

        plt.figure(figsize=(10, 7))
        plt.plot(avg)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Return", fontsize=22)
        plt.savefig('ddqn_policy_set3_32r_16sync_{}.png'.format(size))
