import os
import sys
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/mininet')        
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages')                                                           

import time
import numpy as np
import socket
import threading

import proactive_paths_computation
import proactive_topology_mininet

# active_paths = {}
paths = {}
controller_stats = {}


class MininetAPI(object):    
    def __init__(self, n_hosts, n_paths):
        global paths
        
        self.n_hosts = n_hosts
        self.n_paths = n_paths
        
        # build networkx graph
        self.G = proactive_paths_computation.build_graph_from_txt()
        
        # build mininet net
        self.net = proactive_topology_mininet.start_network()
        
        # get K-shortest paths between each hosts pair
        paths = proactive_paths_computation.get_k_shortest_paths(self.G, self.n_hosts, self.n_paths)
        
        # socket to receive switch stats from the controller 
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(('127.0.0.1', 6631))
        
        self.t = threading.Thread(target=self.read_from_socket)
        self.t.start()
    
    # fill the controller_stats dict from socket
    def read_from_socket(self):
        global controller_stats
        
        while True:
            size = int.from_bytes(self.s.recv(4), 'little')
            data = self.s.recv(size)
            decoded_data = data.decode('utf-8')
            
            items = decoded_data.split("/")
            for item in items:
                if len(item) > 0:
                    elements = item.split("_")
                    controller_stats[(elements[0], elements[1])] = elements[2]
                    
            # print(controller_stats)
    
    # build the network state using the controller stats and paths dict
    def build_state(self):
        state = np.empty((self.n_hosts,self.n_hosts,self.n_paths,1), dtype=object)
        
        for src in range(1, self.n_hosts+1):
            h_src = "H{}".format(src)
            for dst in range(1, self.n_hosts+1):
                h_dst = "H{}".format(dst)
                min_value = float('Inf')
                cnt = 0
                if len(paths[(h_src, h_dst)]) == 1:
                    if paths[(h_src, h_dst)] == []:
                        for idx in range(self.n_paths):
                            state[src-1, dst-1, idx] = 1
                    else: 
                        state[src-1, dst-1, 0] = 102400
                        for idx in range(1, self.n_paths):
                            state[src-1, dst-1, idx] = 1
                else:
                    for path in paths[(h_src, h_dst)]:
                        path = path[1:-1]
                        for s1, s2 in zip(path[:-1], path[1:]):
                            stats = controller_stats.get((str(s1), str(s2)))
                            if stats:
                                if float(stats) < float(min_value):
                                    min_value = controller_stats[(str(s1), str(s2))]
                    
                        state[src-1, dst-1, cnt] = float(min_value)
                        cnt += 1
                        
                    for idx in range(len(paths[(h_src, h_dst)]), self.n_paths):
                        state[src-1, dst-1, idx] = 1
                    
        return state
    
    # send paths to the controller for rule installation
    def send_path_to_controller(self, action, server, client):
        path = paths[(client, server)][action]
        path_r = paths[(server, client)][action]
             
        try: 
            f = open("active_paths.txt", "a")
            f.write('{}_{}_{} \n'.format(server, client, path_r))
            f.write('{}_{}_{} \n'.format(client, server, path))
            f.close()
        except IOError:
            print("file not ready")   
        
        time.sleep(1)
     
    # start traffic flows with iperf
    def start_iperf(self, action, request_number):             
        if request_number == 0:
            self.send_path_to_controller(action, 'H4', 'H1')
            dst_ip = self.net.getNodeByName('H4').IP()
            self.net.getNodeByName('H4').cmd('iperf3 -s -i 1 -p 1111 >& h4_server_1.log &')
            self.net.getNodeByName('H1').cmd('iperf3 -c {} -u -b 30M -t 20 -p 1111 >& h1_h4_client1.log &'.format(dst_ip))
        
        if request_number == 1:
            self.send_path_to_controller(action, 'H5', 'H1')
            dst_ip = self.net.getNodeByName('H5').IP()
            self.net.getNodeByName('H5').cmd('iperf3 -s -i 1 -p 2222 >& h5_server_2.log &')
            self.net.getNodeByName('H1').cmd('iperf3 -c {} -u -b 30M -t 20 -p 2222 >& h1_h5_client2.log &'.format(dst_ip))
        
        if request_number == 2:
            self.send_path_to_controller(action, 'H8', 'H1')
            dst_ip = self.net.getNodeByName('H8').IP()
            self.net.getNodeByName('H8').cmd('iperf3 -s -i 1 -p 3333 >& h8_server_3.log &')
            self.net.getNodeByName('H1').cmd('iperf3 -c {} -u -b 30M -t 20 -p 3333 >& h1_h8_client3.log &'.format(dst_ip))
        
    # clear files and variables
    def reset_measures(self):
        os.system("rm -f ./*.log")
        open('active_paths.txt', 'w').close()  
