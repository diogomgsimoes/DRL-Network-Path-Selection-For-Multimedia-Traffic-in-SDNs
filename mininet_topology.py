#!/usr/bin/python         

import sys
sys.path.insert(0, '/home/dmg/Desktop/DRLResearch/mininet')                                                                   

from mininet.net import Mininet
from mininet.cli import CLI           
from mininet.util import dumpNodeConnections                                             
from mininet.log import setLogLevel, info
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink, Link


class MininetAPI(object):
    def __init__(self, link_bw):
        net = Mininet(controller=RemoteController, link=TCLink, switch=OVSSwitch)

        h1 = net.addHost('h1', mac="00:00:00:00:00:01")
        h2 = net.addHost('h2', mac="00:00:00:00:00:02")
        h3 = net.addHost('h3', mac="00:00:00:00:00:03")
        h4 = net.addHost('h4', mac="00:00:00:00:00:04")
        h5 = net.addHost('h5', mac="00:00:00:00:00:05")
        h6 = net.addHost('h6', mac="00:00:00:00:00:06")
        h7 = net.addHost('h7', mac="00:00:00:00:00:07")

        s1 = net.addSwitch('s1', dpid="1")
        s2 = net.addSwitch('s2', dpid="2")
        s3 = net.addSwitch('s3', dpid="3")
        
        c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)

        linkopt1=dict(bw=100, delay='1ms', loss=0)
        linkopt2=dict(bw=100, delay='1ms', loss=0)
        linkopt3=dict(bw=100, delay='1ms', loss=0)

        net.addLink(h1, s1, **linkopt3)
        net.addLink(h2, s1, **linkopt3)
        net.addLink(h4, s1, **linkopt3)
        net.addLink(h5, s1, **linkopt3)
        net.addLink(h6, s1, **linkopt3)
        net.addLink(h7, s1, **linkopt3)
        net.addLink(h3, s2, **linkopt3)
        net.addLink(s1, s2, **linkopt1)
        net.addLink(s1, s3, **linkopt1)
        net.addLink(s2, s3, **linkopt2)
        
        net.build()
        c0.start()
        s1.start([c0])
        s2.start([c0])
        s3.start([c0])
        
        CLI(net)
        
        # h3.popen('iperf3 -s -i 1 -p 5555', shell=True)
        # h3.popen('iperf3 -s -i 1 -p 6666', shell=True)
        # h3.popen('iperf3 -s -i 1 -p 7777', shell=True)
        # h3.popen('iperf3 -s -i 1 -p 8888', shell=True)
        # h1.cmdPrint('iperf3 -c 10.0.0.3 -u -b 60M -t 50 -p 5555 > h1.log &')
        # h2.cmdPrint('iperf3 -c 10.0.0.3 -u -b 20M -t 50 -p 6666 > h2.log &')
        # h4.cmdPrint('iperf3 -c 10.0.0.3 -u -b 10M -t 50 -p 7777 > h4.log &')
        # h5.cmdPrint('iperf3 -c 10.0.0.3 -u -b 10M -t 50 -p 8888 > h5.log')

        net.stop()

 
if __name__ == '__main__':
    setLogLevel( 'info' )
    MininetAPI(10)   