#!/usr/bin/python                                                                            

import os
import time
from mininet.cli import CLI
from mininet.net import Mininet       
from mininet.topo import Topo              
from mininet.util import dumpNodeConnections                                             
from mininet.log import setLogLevel, info
from mininet.node import Controller, Host, OVSKernelSwitch
from mininet.link import TCLink, Link


class MininetAPI(object):
    def __init__(self, link_bw):
        self.init_params(link_bw)
        self.net = Mininet(topo=None, build=False, listenPort=6633, ipBase='10.0.0.0/8')

        self.c0 = self.net.addController(name='c0', controller=Controller, ip='127.0.0.1', port=6633)

        self.sw1 = self.net.addSwitch('sw1', cls=OVSKernelSwitch)
        self.sw2 = self.net.addSwitch('sw2', cls=OVSKernelSwitch)

        self.h1 = self.net.addHost('h1', cls=Host, ip='10.0.0.1', mac='00:00:00:00:00:01')
        self.h2 = self.net.addHost('h2', cls=Host, ip='10.0.0.2', mac='00:00:00:00:00:02')

        self.net.addLink(self.h1, self.sw1, cls=Link)
        self.net.addLink(self.h2, self.sw2, cls=Link)
        self.net.addLink(self.sw1, self.sw2, cls=TCLink, bw=self.link_bw)

        self.net.build()
        self.net.start()

        self.sw1.cmd('ovs-ofctl --protocols=OpenFlow13 add-flow sw1 priority=10,ip,nw_dst=10.0.0.1,actions=output:1')
        self.sw1.cmd('ovs-ofctl --protocols=OpenFlow13 add-flow sw1 priority=10,arp,nw_dst=10.0.0.1,actions=output:1')
        self.sw1.cmd('ovs-ofctl --protocols=OpenFlow13 add-flow sw1 priority=10,arp,nw_dst=10.0.0.2,actions=normal')
        
        self.h2.cmd("iperf3 -s -i 1 >& /tmp/tcp_server.log &")
        
        #CLI(self.net)
        #self.test()
        #self.get_measures()
        #self.net.stop()
    
    def init_params(self, link_bw):
        self.link_bw = float(link_bw)

    def reset_links(self):
        self.measure()

    def test(self):
        self.net.pingAll()
        self.net.iperf()

    def get_measures(self):
        os.system("rm /tmp/*.log")

        ip = self.h2.IP()
        #udp rn
        cmd = "iperf3 -c {0} -u -t 5 >& /tmp/tcp_client.log &".format(ip)
        self.h1.cmd(cmd)

        # wait for the log file to be correctly generated
        time.sleep(15)

        bandwidth = []
        throughput = []
        jitter = [] 
        packet_loss = []
        with open('/tmp/tcp_client.log') as f:
            for row in f:
                if 'bits/sec' in row:
                    row = row.replace('-', ' ')
                    fields = row.strip().split()
                    if len(fields) > 11:
                        bandwidth.append(fields[7])
                        throughput.append((float)(fields[5])/5)
                        jitter.append(fields[9])
                        processed_packet_loss = fields[12].replace('(', '')
                        processed_packet_loss = processed_packet_loss.replace(')', '')
                        processed_packet_loss = processed_packet_loss.replace('%', '')
                        packet_loss.append(processed_packet_loss)

        print("Bandwidth: " + bandwidth[-1])
        print("Throughput: %.2f" % throughput[-1])
        print("Jitter: " + jitter[-1])
        print("Packet loss: " + packet_loss[-1])
        
        return bandwidth[-1], throughput[-1], jitter[-1], packet_loss[-1]

    def clear(self):
        self.net.stop()


# if __name__ == '__main__':
#     setLogLevel('info')
#     MininetAPI(10)