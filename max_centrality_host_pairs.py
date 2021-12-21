import copy
import networkx as nx
from itertools import islice

NUMBER_OF_HOSTS = 13
NUMBER_OF_PATHS = 5
TOPOLOGY_ARPANET_FILE_NAME = "topology_ARPANET.txt"
hosts = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13']


class DRLEngine():
    def __init__(self):
        self.graph = nx.Graph()
        self.link_bw_capacity = {}
        self.current_link_bw = {}
        self.hosts = {}
        self.paths = {}

        self.upload_topology()
        self.build_graph()
        self.calculate_paths()

    def upload_topology(self):
        with open(TOPOLOGY_ARPANET_FILE_NAME, 'r') as topo:
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
        with open(TOPOLOGY_ARPANET_FILE_NAME, 'r') as topo:
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


engine = DRLEngine()
edge_centrality = nx.edge_load_centrality(engine.graph)
centrality_host_scores = {}
MAX_CENTRALITY_LINK = max(edge_centrality.values())

for src in hosts:
    for dst in hosts:
        if src != dst:
            for path in engine.paths[(src, dst)]:
                host_pair_score = 0
                if len(path) > 3:
                    for s1, s2 in zip(path[1:], path[:-1]):
                        if "H" not in str(s1):
                            _s1 = "S" + str(s1)
                        else:
                            _s1 = str(s1)
                        if "H" not in str(s2):
                            _s2 = "S" + str(s2)
                        else:
                            _s2 = str(s2)

                        link_centrality = edge_centrality.get((_s1, _s2))
                        host_pair_score += link_centrality / MAX_CENTRALITY_LINK
                    if not centrality_host_scores.get((src, dst)):
                        centrality_host_scores[(src, dst)] = host_pair_score
                    else:
                        centrality_host_scores[(src, dst)] += host_pair_score

for key in list(centrality_host_scores.keys()):
    idx_src = key[0].replace("H", "")
    if int(idx_src) in [6, 8, 9, 10, 11, 12, 13]:
        del centrality_host_scores[key]
        continue

    idx_dst = key[1].replace("H", "")
    if int(idx_dst) in [1, 2, 3, 4, 5, 7]:
        del centrality_host_scores[key]

sorted_dict = sorted(centrality_host_scores.items(), key=lambda x: int(x[1]))
print(sorted_dict)
