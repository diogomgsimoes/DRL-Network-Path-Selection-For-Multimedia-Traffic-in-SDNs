import os
import sys
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/mininet')        
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages')                                                           

import time
import numpy as np
import random
import copy

import proactive_paths_computation
import proactive_topology_mininet

paths = {}
bw = {}
bw_capacity = {}
state_helper = {}
controller_stats = {}
busy_ports = [6631, 6633]
host_pairs = [('H4', 'H8'), ('H2', 'H11'), ('H2', 'H13'), ('H2', 'H9'), ('H4', 'H11'), ('H4', 'H9'), ('H2', 'H8'), ('H1', 'H11'),
             ('H1', 'H9'), ('H4', 'H13'), ('H4', 'H10'), ('H4', 'H7'), ('H3', 'H8'), ('H2', 'H10'), ('H2', 'H7'), ('H1', 'H8'), 
             ('H4', 'H12'), ('H3', 'H11'), ('H2', 'H12'), ('H1', 'H13'), ('H3', 'H9'), ('H1', 'H12'), ('H1', 'H7'), ('H4', 'H6'), 
             ('H3', 'H10'), ('H5', 'H6'), ('H3', 'H13'), ('H3', 'H7'), ('H7', 'H6'), ('H5', 'H11'), ('H5', 'H8'), ('H3', 'H12')]


class MininetAPI(object):    
    def __init__(self, n_hosts, n_paths):
        global paths, bw_capacity, bw
        
        self.n_hosts = n_hosts
        self.n_paths = n_paths
        
        # build networkx graph
        self.G = proactive_paths_computation.build_graph_from_txt()
        
        # build mininet net and install ARP rules
        bw_capacity, self.net = proactive_topology_mininet.start_network()
        bw = copy.deepcopy(bw_capacity)
        self.add_arps()
        
        # get K-shortest paths between each hosts pair
        paths = proactive_paths_computation.get_k_shortest_paths(self.G, self.n_hosts, self.n_paths)
    
    # build the network state using the controller stats and paths dict
    def build_state(self):
        state = np.empty((self.n_hosts, self.n_hosts, self.n_paths, 1), dtype=object)
        
        for src in range(1, self.n_hosts+1):
            h_src = "H{}".format(src)
            for dst in range(1, self.n_hosts+1):
                h_dst = "H{}".format(dst)
                min_value = float('Inf')
                cnt = 0
                if len(paths[(h_src, h_dst)]) == 1:
                    if not paths[(h_src, h_dst)]:
                        for idx in range(self.n_paths):
                            state[src-1, dst-1, idx] = 1
                    else: 
                        state[src-1, dst-1, 0] = 100
                        for idx in range(1, self.n_paths):
                            state[src-1, dst-1, idx] = 1
                else:
                    for path in paths[(h_src, h_dst)]:
                        path = path[1:-1]
                        for s1, s2 in zip(path[:-1], path[1:]):
                            if "S" not in str(s1):
                                _s1 = "S" + str(s1)
                            if "S" not in str(s2):
                                _s2 = "S" + str(s2)
                            stats = bw.get((str(_s1), str(_s2)))
                            if stats:
                                if float(stats) < float(min_value):
                                    min_value = bw[(str(_s1), str(_s2))]
                                    state_helper[str(src) + "_" + str(dst) + "_" + str(cnt)] = str(s1) + "_" + str(s2)
                        
                        state[src-1, dst-1, cnt] = float(min_value)
                        cnt += 1
                        
                    for idx in range(len(paths[(h_src, h_dst)]), self.n_paths):
                        state[src-1, dst-1, idx] = 1
                    
        return state
    
    def get_percentage(self, src, dst, bw):
        src_str = "S" + str(src)
        dst_str = "S" + str(dst)
        if bw_capacity.get((src_str, dst_str)):
            return (bw / bw_capacity.get((src_str, dst_str))) * 100
        else:
            return None

    def get_state_helper(self):
        return state_helper
    
    # send paths to the controller for rule installation
    def send_path_to_controller(self, action, client, server):
        global bw
        
        path = paths[(client, server)][action]
        path_r = paths[(server, client)][action]
             
        try: 
            f = open("drl_active_paths.txt", "a")
            f.write('{}_{}_{} \n'.format(server, client, path_r))
            f.write('{}_{}_{} \n'.format(client, server, path))
            f.close()
        except IOError:
            print("file not ready")   
        
        time.sleep(1)
        
        for idx in range(len(path)):
            if 'H' not in str(path[idx]):
                path[idx] = "S" + str(path[idx])
        
        _path = path[1:-1]
        for s1, s2 in zip(_path[:-1], _path[1:]):
            if bw.get((str(s1), str(s2))):
                bw[(str(s1), str(s2))] -= 15
                if bw[(str(s1), str(s2))] == 0:
                    bw[(str(s1), str(s2))] = 1
            if bw.get((str(s2), str(s1))):
                bw[(str(s2), str(s1))] -= 15
                if bw[(str(s2), str(s1))] == 0:
                    bw[(str(s2), str(s1))] = 1  
     
    # start traffic flows with iperf
    def start_iperf(self, action):
        hosts_pair = host_pairs.pop(0)
        self.send_path_to_controller(action, hosts_pair[0], hosts_pair[1])
        
        while True:
            port = random.randint(1000, 9999)
            if port not in busy_ports: break

        dst_ip = self.net.getNodeByName(hosts_pair[1]).IP()
        self.net.getNodeByName(hosts_pair[1]).cmd('iperf3 -s -i 1 -p {} >& {}_server_{}.log &'.format(port, hosts_pair[1], port))
        self.net.getNodeByName(hosts_pair[0]).cmd('iperf3 -c {} -J -b 15M -t 180 -p {} >& {}_{}_client_{}.log &'.format(dst_ip, port, hosts_pair[0], hosts_pair[1], port))
    
    # define starting ARP rules
    def add_arps(self):
        for id_src in range(1, 14):
            for id_dst in range(1, 14):
                dst_mac = "00:00:00:00:00:{}".format(str(id_dst).zfill(2))
                
                dst_mac_int = int(dst_mac[-2:])
                dst_mac_hex = "{:012x}".format(dst_mac_int)
                dst_mac_hex_str = ":".join(dst_mac_hex[i:i+2] for i in range(0, len(dst_mac_hex), 2))
        
                self.net.getNodeByName("H" + str(id_src)).cmd("arp -s 10.0.0.{} {}".format(id_dst, dst_mac_hex_str))
            
    # clear files and variables
    def reset_measures(self):
        global busy_ports, host_pairs, bw
        
        bw = copy.deepcopy(bw_capacity)
        
        os.system("rm -f ./*.log")
        open('drl_active_paths.txt', 'w').close()  
        busy_ports = [6631, 6633]
        host_pairs = [('H4', 'H8'), ('H2', 'H11'), ('H2', 'H13'), ('H2', 'H9'), ('H4', 'H11'), ('H4', 'H9'), ('H2', 'H8'), ('H1', 'H11'),
             ('H1', 'H9'), ('H4', 'H13'), ('H4', 'H10'), ('H4', 'H7'), ('H3', 'H8'), ('H2', 'H10'), ('H2', 'H7'), ('H1', 'H8'), 
             ('H4', 'H12'), ('H3', 'H11'), ('H2', 'H12'), ('H1', 'H13'), ('H3', 'H9'), ('H1', 'H12'), ('H1', 'H7'), ('H4', 'H6'), 
             ('H3', 'H10'), ('H5', 'H6'), ('H3', 'H13'), ('H3', 'H7'), ('H7', 'H6'), ('H5', 'H11'), ('H5', 'H8'), ('H3', 'H12')]
