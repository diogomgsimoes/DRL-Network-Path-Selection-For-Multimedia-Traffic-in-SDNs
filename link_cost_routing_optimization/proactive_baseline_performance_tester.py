import os
import threading
import time
import random
import copy
from datetime import datetime

import proactive_topology_mininet

active_comms = []
last_comms = []
shortest_paths = {}
busy_ports = [6631, 6633]
host_pairs = [('H4', 'H8'), ('H2', 'H11'), ('H2', 'H13'), ('H2', 'H9'), ('H4', 'H11'), ('H4', 'H9'), ('H2', 'H8'), ('H1', 'H11'),
           ('H1', 'H9'), ('H4', 'H13'), ('H4', 'H10'), ('H4', 'H7'), ('H3', 'H8'), ('H2', 'H10'), ('H2', 'H7'), ('H1', 'H8'),
           ('H4', 'H12'), ('H3', 'H11'), ('H2', 'H12'), ('H1', 'H13'), ('H3', 'H9'), ('H1', 'H12'), ('H1', 'H7'), ('H4', 'H6'), 
           ('H3', 'H10'), ('H5', 'H6'), ('H3', 'H13'), ('H3', 'H7'), ('H7', 'H6'), ('H5', 'H11'), ('H5', 'H8'), ('H3', 'H12')]


def simulate(net):
    global host_pairs
    
    hosts_pair = host_pairs.pop(0)
    
    while True:
        port = random.randint(1000, 9999)
        if port not in busy_ports: break
        
    thread = threading.Thread(target=set_active_paths, args=(hosts_pair,))
    thread.start()
    
    time.sleep(5)

    dst_ip = net.getNodeByName(hosts_pair[1]).IP()
    net.getNodeByName(hosts_pair[1]).cmd('iperf3 -s -i 1 -p {} >& {}_server_{}.log &'.format(port, hosts_pair[1], port))
    time.sleep(0.1)
    net.getNodeByName(hosts_pair[0]).cmd('iperf3 -c {} -b 15M -J -t 90 -p {} >& {}_{}_client_{}.log &'.format(dst_ip, port, hosts_pair[0], hosts_pair[1], port))
    
def set_active_paths(pair):
    global active_comms
    active_comms.append(pair) 
    time.sleep(90)
    active_comms.remove(pair)
    print("iperf ended for ", pair)

def transfer_active_paths():
    global active_comms, last_comms
    while True:
        if last_comms != active_comms:
            try: 
                f = open("baseline_active_paths.txt", "w")
                for item in active_comms:
                    f.write('{}_{}\n'.format(item[0], item[1]))
                f.close()
            except IOError:
                print("file not ready")   
                
            last_comms = copy.deepcopy(active_comms)
            
# Proactively install ARP rules to avoid ARP flooding
def add_arps(net):
    for id_src in range(1, 14):
        for id_dst in range(1, 14):
            dst_mac = "00:00:00:00:00:{}".format(str(id_dst).zfill(2))
            
            dst_mac_int = int(dst_mac[-2:])
            dst_mac_hex = "{:012x}".format(dst_mac_int)
            dst_mac_hex_str = ":".join(dst_mac_hex[i:i+2] for i in range(0, len(dst_mac_hex), 2))
    
            net.getNodeByName("H" + str(id_src)).cmd("arp -s 10.0.0.{} {}".format(id_dst, dst_mac_hex_str))

def clear_structures():
    os.system("rm -f ./*.log")
    open('baseline_active_paths.txt', 'w').close()  


if __name__ == "__main__":
    _, net = proactive_topology_mininet.start_network()
    
    clear_structures()
    add_arps(net)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    
    update_thread = threading.Thread(target=transfer_active_paths)
    update_thread.start()
    
    while host_pairs:
        simulate(net)
