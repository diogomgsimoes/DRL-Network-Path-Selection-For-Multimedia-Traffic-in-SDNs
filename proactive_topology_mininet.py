import sys
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/mininet')                                                                   

from mininet.net import Mininet
from mininet.cli import CLI         
from mininet.log import setLogLevel                                           
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink


TOPOLOGY_FILE_NAME = 'topology_arpanet.txt'


bw_capacity = {}


def add_host(net, host):
    if host not in net.keys():
        net.addHost(host)

def add_switch(net, switch):
    if switch not in net.keys():
        net.addSwitch(switch, cls=OVSSwitch) 

def add_link(net, src, dst, link_bw):
    if len(net.linksBetween(net.getNodeByName(src), net.getNodeByName(dst))) == 0:
        net.addLink(net.getNodeByName(src), net.getNodeByName(dst), **dict(bw=link_bw, delay='1ms', loss=0))

def build_topology_from_txt(net):
    global bw_capacity
    
    with open(TOPOLOGY_FILE_NAME, 'r') as topo:
        for row in topo.readlines():
            row_data = row.split()
            for node in row_data:
                if 'H' in node:
                    add_host(net, node)
                elif 'S' in node:
                    add_switch(net, node)
            link_bw = int(row_data[2])
            add_link(net, row_data[0], row_data[1], link_bw)
            bw_capacity[(row_data[0], row_data[1])] = link_bw
            bw_capacity[(row_data[1], row_data[0])] = link_bw

def start_network():
    setLogLevel('info')
    
    net = Mininet(controller=RemoteController, switch=OVSSwitch, link=TCLink, autoSetMacs=True, ipBase='10.0.0.0/24')
    
    build_topology_from_txt(net)
    net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)
    net.start()
    
    return bw_capacity, net
    
if __name__ == '__main__':
    bw_capacity, net = start_network()
    
    CLI(net)
    net.stop()
