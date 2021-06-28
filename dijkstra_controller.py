import sys
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages')

topo_switches = {1: {1: 0, 2: 10, 3: 20}, 2: {
    1: 10, 2: 0, 3: 50}, 3: {1: 20, 2: 50, 3: 0}}
_switches = set(topo_switches)

def draw_topology():
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.set_title('Network topology', fontsize=10)

    G = nx.DiGraph()
    G.add_node('h1')
    G.add_node('h2')
    G.add_node('h3')
    G.add_node('s1')
    G.add_node('s2')
    G.add_node('s3')
    G.add_edge('s1', 's2', label='10')
    G.add_edge('s2', 's3', label='50')
    G.add_edge('s1', 's3', label='20')
    G.add_edge('h1', 's1', label='1')
    G.add_edge('h2', 's1', label='1')
    G.add_edge('s2', 'h3', label='1')
    pos = nx.spring_layout(G) 
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G,'label'))
    plt.savefig("graph.png")


def minimum_distance(distance, neighbours):
    min = float('inf')
    closest_node = 0
    for node in neighbours:
        if distance[node] < min:
            min = distance[node]
            closest_node = node
    return closest_node


def dijkstra(graph, src, dest):
    dst = dest
    print("Searching for the shortest path between %s and %s..." % (src, dst))
    graph_keys = set(graph)

    distance = {}
    previous = {}

    for dpid in graph_keys:
        distance[dpid] = float('Inf')
        previous[dpid] = None

    distance[src] = 0

    while len(graph_keys) > 0:
        closest_node = minimum_distance(distance, graph_keys)
        graph_keys.remove(closest_node)

        for node in graph_keys:
            if graph[closest_node][node] != None:
                if distance[node] > distance[closest_node] + graph[closest_node][node]:
                    distance[node] = distance[closest_node] + \
                        graph[closest_node][node]
                    previous[node] = closest_node

    path = []
    path.append(dst)
    current_node = previous[dst]

    while current_node is not None:
        if current_node == src:
            path.append(current_node)
            break

        dst = current_node
        path.append(dst)
        current_node = previous[dst]

    path.reverse()

    if src == dst:
        path = [src]
        distance[src] = graph[src][dst]

    print("Shortest path between ", src, " and ", dest, " is: ",
          path, " and the total distance is: ", distance[dest])


def main():
    draw_topology()
    for src in _switches:
        for dst in _switches:
            dijkstra(topo_switches, src, dst)


if __name__ == "__main__":
    main()