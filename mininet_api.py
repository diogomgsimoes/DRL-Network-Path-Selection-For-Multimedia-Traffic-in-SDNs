#!/usr/bin/python         

import os
import sys

sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/mininet')        
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/thesis_env/lib/python3.8/site-packages')                                                           

from mininet.net import Mininet
from mininet.cli import CLI           
from mininet.util import dumpNodeConnections                                             
from mininet.log import setLogLevel, info
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink, Link
import numpy as np
import time
from collections import defaultdict
import json
import networkx as nx
from itertools import islice
from random import randint, choice

# file_n_possibilities = list(range(1, 11))
# port_possibilities = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1333]

paths = {}

def get_paths():
    return paths

class MininetAPI(object):

    def init_params(self, min, max, link_bw):
        self.min = float(min)
        self.max = float(max)
        self.link_bw = float(link_bw)
        self.number_hosts = 3
        
        self.G = nx.DiGraph()
    
        self.G.add_node('h1')
        self.G.add_node('h2')
        self.G.add_node('h3')
        self.G.add_node('s1')
        self.G.add_node('s2')
        self.G.add_node('s3')
        self.G.add_edge('s1', 's2')
        self.G.add_edge('s2', 's1')
        self.G.add_edge('s2', 's3')
        self.G.add_edge('s3', 's2')
        self.G.add_edge('s1', 's3')
        self.G.add_edge('s3', 's1')
        self.G.add_edge('h1', 's1')
        self.G.add_edge('s1', 'h1')
        self.G.add_edge('h2', 's1')
        self.G.add_edge('s1', 'h2')
        self.G.add_edge('s2', 'h3')
        self.G.add_edge('h3', 's2')
    
    
    def __init__(self, min, max, link_bw):
        self.init_params(min, max, link_bw)
               
        self.net = Mininet(controller=RemoteController, link=TCLink, switch=OVSSwitch)

        self.h1 = self.net.addHost('h1', mac="00:00:00:00:00:01")
        self.h2 = self.net.addHost('h2', mac="00:00:00:00:00:02")
        self.h3 = self.net.addHost('h3', mac="00:00:00:00:00:03")
        self.h4 = self.net.addHost('h4', mac="00:00:00:00:00:04")
        self.h5 = self.net.addHost('h5', mac="00:00:00:00:00:05")
        self.h6 = self.net.addHost('h6', mac="00:00:00:00:00:06")
        self.h7 = self.net.addHost('h7', mac="00:00:00:00:00:07")
        self.h8 = self.net.addHost('h8', mac="00:00:00:00:00:08")
        self.h9 = self.net.addHost('h9', mac="00:00:00:00:00:09")

        self.s1 = self.net.addSwitch('s1', dpid="1")
        self.s2 = self.net.addSwitch('s2', dpid="2")
        self.s3 = self.net.addSwitch('s3', dpid="3")
        
        self.c0 = self.net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)

        linkopt1 = dict(bw=self.link_bw, delay='1ms', loss=0)
        linkopt2 = dict(bw=self.link_bw, delay='1ms', loss=0)
        linkopt3 = dict(bw=self.link_bw, delay='1ms', loss=0)

        self.net.addLink(self.h1, self.s1, **linkopt3)
        self.net.addLink(self.h2, self.s1, **linkopt3)
        self.net.addLink(self.h4, self.s1, **linkopt3)
        self.net.addLink(self.h5, self.s1, **linkopt3)
        self.net.addLink(self.h6, self.s1, **linkopt3)
        self.net.addLink(self.h7, self.s1, **linkopt3)
        self.net.addLink(self.h8, self.s1, **linkopt3)
        self.net.addLink(self.h9, self.s1, **linkopt3)
        self.net.addLink(self.h3, self.s2, **linkopt3)
        self.net.addLink(self.s1, self.s2, **linkopt1)
        self.net.addLink(self.s1, self.s3, **linkopt1)
        self.net.addLink(self.s2, self.s3, **linkopt2)
        
        self.net.build()
        self.c0.start()
        self.s1.start([self.c0])
        self.s2.start([self.c0])
        self.s3.start([self.c0])
        
        
    def build_action_space(self, number_paths):
        action_space = defaultdict(lambda: defaultdict(lambda: None))
        
        for src_host_id in range(1, self.number_hosts+1):
            src = "h{}".format(src_host_id)
            for dst_host_id in range(1, self.number_hosts+1):
                dst = "h{}".format(dst_host_id)
                action_space[src][dst] = self.k_shortest_paths(src, dst, number_paths)
                
        return action_space
            
    def k_shortest_paths(self, source, target, k, weight=None):
        try: 
            calc = list(islice(nx.shortest_simple_paths(self.G, source, target, weight=weight), k))
        except nx.NetworkXNoPath:
            calc = []
            
        return [path for path in calc]
         
    def build_state(self):
        state = defaultdict(lambda: defaultdict(lambda: None))
        print("Collecting data...")
        for filename in os.listdir('.'):
            if filename.endswith('.log') and "client" in filename:
                with open(filename) as f:
                    _filename = filename.replace("client", "")
                    _filename = _filename.replace(".log", "")
                    _edges = _filename.split("_")
                    for row in f:
                        if 'bits/sec' in row:
                            row = row.replace('-', ' ')
                            fields = row.strip().split()
                            if len(fields) > 11 and float(fields[7]) != float(30):
                                # bitrate: 7
                                # bandwidth: 5
                                # packet loss: 12 (replace '(', ')' and '%')
                                # jitter: 9
                                src = int(_edges[0].replace("h", ""))
                                dst = int(_edges[1].replace("h", ""))
                                if state[src][dst] != None:
                                    state[src][dst].append(float(fields[7]))
                                    #state[dst][src].append(float(fields[7]))
                                else: 
                                    state[src][dst] = [float(fields[7])]
                                    #state[dst][src] = [float(fields[7])]
                                    
                                # TODO: decide between hosts and switches
                                # need to add the path
              
        print(json.dumps(state))                                     
        return state
        
    def start_iperf(self):
        # src, dst
        # global port_possibilities, file_n_possibilities
        
        # bandwidth = randint(self.min, self.max)
        # src_address = self.net.get(src)
        # dst_address = self.net.get(dst)
        # port = choice(port_possibilities)
        # port_possibilities.remove(port)
        # file_key = choice(file_n_possibilities)
        # file_n_possibilities.remove(file_key)
        # ip = dst_address.IP()
        
        # print(bandwidth, src_address, dst_address, port, file_key, ip)
        
        # dst_address.cmd('iperf3 -s -i 1 -p {} >& {}_server{}.log &'.format(port, src, file_key))
        # src_address.cmd('iperf3 -c {} -u -b 30M -t 30 -p {} >& {}_{}_client{}.log &'.format(ip, port, src, dst, file_key))
        
        # paths should come from mininet_env action space
        paths["3_1"] = [2, 3, 1]
        
        try: 
            f = open("paths.txt", "w")
            f.write("3 1 [2,3,1] \n")
            f.write("1 3 [1,3,2] \n")
            f.close()
        except IOError:
            print("file not ready")   
        
        time.sleep(2)
        
        ip = self.h3.IP()
        
        self.h3.cmd('iperf3 -s -i 1 -p 1111 >& h3_server_1.log &')
        
        # time.sleep(1)
        self.h1.cmd('iperf3 -c {} -u -b 30M -t 30 -p 1111 >& h1_h3_client1.log &'.format(ip))
        
        # time.sleep(5)
        
        # self.h3.cmd('iperf3 -s -i 1 -p 2222 >& h3_server_2.log &')
        # time.sleep(1)
        # self.h2.cmd('iperf3 -c {} -u -b 30M -t 30 -p 2222 >& h2_h3_client2.log &'.format(ip))
        
        # time.sleep(5)
        
        # self.h3.cmd('iperf3 -s -i 1 -p 3333 >& h3_server_3.log &')
        # time.sleep(1)
        # self.h4.cmd('iperf3 -c {} -u -b 30M -t 30 -p 3333 >& h4_h3_client3.log &'.format(ip))
        
        # time.sleep(5)
        
        # self.h3.cmd('iperf3 -s -i 1 -p 4444 >& h3_server_4.log &')
        # time.sleep(1)
        # self.h5.cmd('iperf3 -c {} -u -b 30M -t 30 -p 4444 >& h5_h3_client4.log &'.format(ip))
        
        # time.sleep(5)
        
        # self.h3.cmd('iperf3 -s -i 1 -p 5555 >& h3_server_5.log &')
        # time.sleep(1)
        # self.h6.cmd('iperf3 -c {} -u -b 30M -t 30 -p 5555 >& h6_h3_client5.log &'.format(ip))
        
        # TODO: communicate with the controller to declare the correct path for the flow
        
    def reset_measures(self):
        global port_possibilities, file_n_possibilities
        
        file_n_possibilities = list(range(1, 11))
        port_possibilities = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1333]
        
        os.system("rm -f ./*.log")
        
        #time.sleep(10)
