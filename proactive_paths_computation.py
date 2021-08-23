import sys   
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages') 

import networkx as nx
from itertools import islice

TOPOLOGY_FILE_NAME = 'topology.txt'


def build_graph_from_txt(weights=None):
    graph = nx.Graph()
    
    with open(TOPOLOGY_FILE_NAME, 'r') as topo:
        for line in topo.readlines():
            nodes = line.split()[:2]
            for node in nodes:
                if not graph.has_node(node):
                    graph.add_node(node)
            if 'S' in nodes[0] and 'S' in nodes[1]:
                if weights:
                    sw1 = nodes[0].replace("S", "")
                    sw2 = nodes[1].replace("S", "")
                    graph.add_edge(nodes[0], nodes[1], weight=weights[(sw1, sw2)])
                    graph.add_edge(nodes[1], nodes[0], weight=weights[(sw2, sw1)])
                else:
                    graph.add_edge(nodes[0], nodes[1], weigth=1000000)
                    graph.add_edge(nodes[1], nodes[0], weigth=1000000)
            else:
                graph.add_edge(nodes[0], nodes[1], weigth=1000000)
                graph.add_edge(nodes[1], nodes[0], weigth=1000000)
            
    return graph

def get_k_shortest_paths(graph, number_hosts, number_paths):
    paths = {}
    
    for src_host_id in range(1, number_hosts+1):
        src = "H{}".format(src_host_id)
        for dst_host_id in range(1, number_hosts+1):
            dst = "H{}".format(dst_host_id)
            paths[(src, dst)] = k_shortest_paths(graph, src, dst, number_paths)
            for path in paths[(src, dst)]:
                if len(path) != 0:
                    for i in range(0, len(path)):
                        path[i] = path[i].replace("S", "")
                        path[i] = int(path[i])
                    path.append(dst)
                    path.insert(0, src)
            
    return paths
            
def k_shortest_paths(graph, source, target, k):
    try: 
        calc = list(islice(nx.shortest_simple_paths(graph, source, target), k))
    except nx.NetworkXNoPath:
        calc = []
        
    return [path[1:-1] for path in calc]

def get_src_dst_names(src_mac, dst_mac):
    src = "H" + src_mac[-2:]
    if src[-1] != "0" and "0" in src:
        src = src.replace("0", "")
        
    dst = "H" + dst_mac[-2:]
    if dst[-1] != "0" and "0" in dst:
        dst = dst.replace("0", "")
        
    return (src, dst)

def dijkstra_from_macs(graph, src_mac, dst_mac, host_to_switch_port, adjacency):
    (src, dst) = get_src_dst_names(src_mac, dst_mac)
        
    path = nx.dijkstra_path(graph, src, dst)
    
    # path_tuples = add_ports_to_path(path, host_to_switch_port, adjacency, src_mac, dst_mac)
    
    return path
    
def add_ports_to_path(path, host_to_switch_port, adjacency, src_mac, dst_mac):
    p_tuples = []
    
    if len(path) < 4:
        p_tuples.append((path[1][1:], host_to_switch_port[src_mac][path[1][1:]], 
                        host_to_switch_port[dst_mac][path[1][1:]]))
        return p_tuples
    
    switches_in_path = path[1:-1]
    p_tuples.append((switches_in_path[0][1:], host_to_switch_port[src_mac][switches_in_path[0][1:]], 
                    adjacency[(switches_in_path[0][1:], switches_in_path[1][1:])]))
    
    for i in range(1, len(switches_in_path)-1):
        p_tuples.append((switches_in_path[i][1:], adjacency[(switches_in_path[i][1:], switches_in_path[i-1][1:])], 
                        adjacency[(switches_in_path[i][1:], switches_in_path[i+1][1:])]))
        
    p_tuples.append((switches_in_path[-1][1:], adjacency[(switches_in_path[-1][1:], switches_in_path[-2][1:])], 
                    host_to_switch_port[dst_mac][switches_in_path[-1][1:]]))
        
    return p_tuples

