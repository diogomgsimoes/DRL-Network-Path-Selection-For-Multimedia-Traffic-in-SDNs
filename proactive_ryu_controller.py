from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib.packet import packet, arp
from collections import defaultdict
from operator import attrgetter
import json
import time

import proactive_paths_computation

TOPOLOGY_FILE_NAME = 'topology.txt'
NUMBER_SWITCHES = 10

host_to_switch_port = defaultdict(lambda: defaultdict(lambda: None))
adjacency = {}
paths = {}
number_flows = {}
costs = {}
byte = {}
clock = {}
bw_used = {}
bw_available = {}
bw = {}
switch_ports = {}
_switches = []
active_paths = {}
inactive_flows = {}
host_ip_mac = {}
active_flows = {}
paths_hops = {}
active_hosts = []


def topology_discovery():
    global adjacency, host_to_switch_port, switch_ports, costs, number_flows, _switches, host_ip_mac
    
    with open(TOPOLOGY_FILE_NAME, 'r') as topo:
        for row in topo.readlines():
            row_data = row.split()
            if 'H' in row_data[0] or 'H' in row_data[1]:
                if 'S' in row_data[1]:
                    host_id = row_data[0].replace("H", "")
                    host_ip = "10.0.0.{}".format(host_id)
                    host_mac_addr = "00:00:00:00:00:{}".format(host_id.zfill(2))
                    host_ip_mac[host_ip] = host_mac_addr
                    switch_id = row_data[1][1:]
                    switch_ports[switch_id] = switch_ports[switch_id] + 1 if switch_ports.get(switch_id) else 1
                    host_to_switch_port[host_mac_addr][switch_id] = switch_ports[switch_id]
                    _switches.append(switch_id) if switch_id not in _switches else None
                elif 'S' in row_data[0]:
                    host_id = row_data[1].replace("H", "")
                    host_ip = "10.0.0.{}".format(host_id)
                    host_mac_addr = "00:00:00:00:00:{}".format(host_id.zfill(2))
                    host_ip_mac[host_ip] = host_mac_addr
                    switch_id = row_data[0][1:]
                    switch_ports[switch_id] = switch_ports[switch_id] + 1 if switch_ports.get(switch_id) else 1
                    host_to_switch_port[host_mac_addr][switch_id] = switch_ports[switch_id]
                    _switches.append(switch_id) if switch_id not in _switches else None
            elif 'S' in row_data[0] and 'S' in row_data[1]:
                sw1 = row_data[0][1:]
                sw2 = row_data[1][1:]
                switch_ports[sw1] = switch_ports[sw1] + 1 if switch_ports.get(sw1) else 1
                switch_ports[sw2] = switch_ports[sw2] + 1 if switch_ports.get(sw2) else 1
                adjacency[(sw1, sw2)] = int(switch_ports[sw1])
                adjacency[(sw2, sw1)] = int(switch_ports[sw2])
                bw[(sw1, sw2)] = float(row_data[2])
                bw[(sw2, sw1)] = float(row_data[2])
                number_flows[(sw1, sw2)] = 0
                number_flows[(sw2, sw1)] = 0
                costs[(sw1, sw2)] = float(0)
                costs[(sw2, sw1)] = float(0)
                _switches.append(sw1) if sw1 not in _switches else None
                _switches.append(sw2) if sw2 not in _switches else None

def load_active_hosts():
    global active_hosts
    
    active_hosts.clear()
    
    try:
        f = open("baseline_active_paths.txt", "r")
        for line in f:
            a = line.strip('\n').split('_')
            if a:   
                src = a[0].replace("H", "")
                src_mac = "00:00:00:00:00:{}".format(src.zfill(2))
                dst = a[1].replace("H", "")
                dst_mac = "00:00:00:00:00:{}".format(dst.zfill(2))

                active_hosts.append((src_mac, dst_mac))
                
        f.close()

    except IOError:
        print("file not ready")

class ProactiveController(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ProactiveController, self).__init__(*args, **kwargs)
        self.monitor_thread = hub.spawn(self._monitor)
        self.datapaths = {}
        
        topology_discovery()
        
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath

        if ev.state == MAIN_DISPATCHER:
            if not datapath.id in self.datapaths:
                print('Datapath {} registered.'.format(datapath.id))
                self.datapaths[datapath.id] = datapath

        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                print('Datapath {} unregistered.'.format(datapath.id))
                del self.datapaths[datapath.id]
                
        if len(self.datapaths) == NUMBER_SWITCHES:
            self.update_paths()
    
    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(3)
    
    def _request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # req = parser.OFPFlowStatsRequest(datapath)
        # datapath.send_msg(req)
        
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()

        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        
        self.add_flow(datapath, 0, match, actions)
        
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        global byte, clock, bw_used, bw_available, costs, paths_hops, stop

        body = ev.msg.body
        dpid = ev.msg.datapath.id

        for stat in sorted(body, key=attrgetter('port_no')):
            for sw in _switches:
                if adjacency.get((str(dpid), sw), 0) == stat.port_no:
                    if int(byte.get((str(dpid), sw), 0)) > 0 and clock.get((str(dpid), sw)):
                        bw_used[(str(dpid), sw)] = (stat.tx_bytes - float(byte.get((str(dpid), sw), 0))) * 8.0 \
                            / (time.time() - float(clock.get((str(dpid), sw), 0))) / 1000
                            
                        bw_available[(str(dpid), sw)] = int(bw.get((str(dpid), sw), 0) \
                            * 1024.0) - float(bw_used.get((str(dpid), sw), 0))
                        
                        # Static version
                        costs[(str(dpid), sw)] = 1
                            
                        # Uncomment for DSP
                        # costs[(str(dpid), sw)] = 1/int(bw_available.get((str(dpid), sw), 0))
                        
                        # if int(number_flows.get((str(dpid), sw), 0)) == 0 and int(number_flows.get((str(dpid), sw), 0)) == 0:
                        #     costs[(str(dpid), sw)] = 1/int(bw_available.get((str(dpid), sw), 0))
                        # else:
                        #     costs[(str(dpid), sw)] = (int(number_flows.get((str(dpid), sw), 0))/2 + \
                        #         int(number_flows.get((str(dpid), sw), 0))/2)/int(bw_available.get((str(dpid), sw), 0))
                            
                    byte[(str(dpid), sw)] = stat.tx_bytes
                    clock[(str(dpid), sw)] = time.time()
                    
        # print("NUMBER_FLOWS:", number_flows)
        # print("COSTS:", costs)
        # if active_hosts:
        #     paths = []
        #     for item in active_hosts:
        #         paths.append(paths_hops.get(item))
        #     if paths:
        #         print("ACTIVE_PATHS:", paths)
        # print("BW_AVAILABLE:", bw_available)
                    
        if len(self.datapaths) == NUMBER_SWITCHES:
            load_active_hosts()
            self.update_paths()
            # self.update_flows()

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        pass
        # global active_flows, number_flows, inactive_flows
        
        # body = ev.msg.body
        
        # for stat in body:
        #     if stat.byte_count > 0:
        #         key = (stat.match.get('eth_src'), stat.match.get('eth_dst'))
        #         if key[0] and key[1]:
        #             active_flow_byte_count = active_flows.get(key)
        #             inactive_flow_byte_count = inactive_flows.get(key)
        #             if active_flow_byte_count:
        #                 if stat.byte_count <= active_flow_byte_count:
        #                     del active_flows[key]
        #                     inactive_flows[key] = stat.byte_count
        #             elif inactive_flow_byte_count:
        #                 if stat.byte_count > inactive_flow_byte_count:
        #                     del inactive_flows[key]
        #                     active_flows[key] = stat.byte_count
        #             else:   
        #                 active_flows[key] = stat.byte_count
                    
        # # print(active_flows)
                
        # number_flows = {}
                
        # for flow in active_flows.items():
        #     hosts_pair = flow[0]
        #     if hosts_pair[0] != None and hosts_pair[1] != None:
        #     # assuming active_paths holds the current paths between all hosts
        #         # print(hosts_pair)
        #         # print("-------------")
        #         curr_path = paths_hops.get(hosts_pair)[1:-1]
        #         dpids_path = [node.replace("S", "") for node in curr_path]
        #         for dpid1, dpid2 in zip(dpids_path[:-1], dpids_path[1:]):
        #             if number_flows.get((dpid1, dpid2)):
        #                 number_flows[(dpid1, dpid2)] += 1
        #             else:
        #                 number_flows[(dpid1, dpid2)] = 1
                        
    def update_flows(self):
        global number_flows
        
        number_flows = {}
        
        for host_pair in active_hosts:
            path_without_hosts = paths_hops.get((host_pair[0], host_pair[1]))[1:-1]
            dpids_path = [node.replace("S", "") for node in path_without_hosts]
            for dpid1, dpid2 in zip(dpids_path[:-1], dpids_path[1:]):
                if number_flows.get((dpid1, dpid2)):
                    number_flows[(dpid1, dpid2)] += 1
                else:
                    number_flows[(dpid1, dpid2)] = 1
                      
    def update_paths(self):
        global paths_hops, paths
        
        for src_host in host_to_switch_port.keys():
            for dst_host in host_to_switch_port.keys():
                if src_host != dst_host:
                    if (src_host, dst_host) not in active_hosts:
                        graph = proactive_paths_computation.build_graph_from_txt(costs)
                        path = proactive_paths_computation.dijkstra_from_macs(
                            graph, src_host, dst_host, host_to_switch_port, adjacency)
                        existing_path = paths_hops.get((src_host, dst_host))
                        if existing_path:
                            if existing_path != path:
                                paths_hops[(src_host, dst_host)] = path
                                self.uninstall_path(paths[(src_host, dst_host)], src_host, dst_host)
                                paths[(src_host, dst_host)] = proactive_paths_computation.add_ports_to_path(
                                    path, host_to_switch_port, adjacency, src_host, dst_host)
                                self.install_path(paths[(src_host, dst_host)], src_host, dst_host)
                        else:
                            paths_hops[(src_host, dst_host)] = path
                            paths[(src_host, dst_host)] = proactive_paths_computation.add_ports_to_path(
                                path, host_to_switch_port, adjacency, src_host, dst_host)
                            self.install_path(paths[(src_host, dst_host)], src_host, dst_host)
 
    def install_path(self, p, src_mac, dst_mac):
        for sw, in_port, out_port in p:
            datapath = self.datapaths.get(int(sw))
            parser = datapath.ofproto_parser
            match = parser.OFPMatch(in_port=in_port, eth_src=src_mac, eth_dst=dst_mac)
            actions = [parser.OFPActionOutput(out_port)]
            self.add_flow(datapath, 1, match, actions)
            
    def uninstall_path(self, p, src_mac, dst_mac):
        for sw, in_port, _ in p:
            datapath = self.datapaths.get(int(sw))
            parser = datapath.ofproto_parser
            match = parser.OFPMatch(in_port=in_port, eth_src=src_mac, eth_dst=dst_mac)
            self.remove_flow(datapath, match)
            
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst, command=ofproto.OFPFC_ADD)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst, command=ofproto.OFPFC_ADD)
            
        datapath.send_msg(mod)
        
    def remove_flow(self, datapath, match):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        mod = parser.OFPFlowMod(datapath=datapath, match=match, priority=1,
                                command=ofproto.OFPFC_DELETE, out_group=ofproto.OFPG_ANY, out_port=ofproto.OFPP_ANY)
            
        datapath.send_msg(mod)
                               
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        ofp_parser = dp.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        pkt = pkt.get_protocol(arp.arp)
        if not pkt: 
            return  
        
        src = pkt.src_mac
        dst = host_ip_mac[pkt.dst_ip]
        dpid = dp.id
        
        out_port = None
        if paths.get((src, dst)) != None:
            path = paths[(src, dst)]
            for sw, _, _out in path:
                if int(sw) == dpid:
                    out_port = _out   
   
        data = None 
        if msg.buffer_id == ofp.OFP_NO_BUFFER:
             data = msg.data

        if out_port:
            actions = [ofp_parser.OFPActionOutput(out_port)]
            out = ofp_parser.OFPPacketOut(
                datapath=dp, buffer_id=msg.buffer_id, in_port=in_port,
                actions=actions, data=data)
            dp.send_msg(out)
