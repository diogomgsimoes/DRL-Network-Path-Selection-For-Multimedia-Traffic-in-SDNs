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
host_pairs = [('H1', 'H4'), ('H1', 'H5'), ('H2', 'H5'), ('H2', 'H8'), ('H3', 'H6'), ('H3', 'H7'), ('H4', 'H5'), ('H4', 'H8')]


def simulate(net):
    global host_pairs
    
    hosts_pair = host_pairs.pop(0)
    
    while True:
        port = random.randint(1000, 9999)
        if port not in busy_ports: break

    dst_ip = net.getNodeByName(hosts_pair[1]).IP()
    net.getNodeByName(hosts_pair[1]).cmd('iperf3 -s -i 1 -p {} >& {}_server_{}.log &'.format(port, hosts_pair[1], port))
    net.getNodeByName(hosts_pair[0]).cmd('iperf3 -c {} -b 20M -t 60 -p {} -J >& {}_{}_client_{}.log &'.format(dst_ip, port, hosts_pair[0], hosts_pair[1], port))
    thread = threading.Thread(target=set_active_paths, args=(hosts_pair,))
    thread.start()
    
    time.sleep(5)
    
def set_active_paths(pair):
    global active_comms
    active_comms.append(pair) 
    time.sleep(60)
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

def clear_structures():
    os.system("rm -f ./*.log")
    open('baseline_active_paths.txt', 'w').close()  


if __name__ == "__main__":
    net = proactive_topology_mininet.start_network()
    
    clear_structures()

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    
    update_thread = threading.Thread(target=transfer_active_paths)
    update_thread.start()
    
    while host_pairs:
        simulate(net)
