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
host_ip_mac = {}
counter = 0
active_paths = {}
last_active_paths_len = 0
ended_flows = 0
p_dict = {}
requested_flows = 0


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


class ProactiveController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ProactiveController, self).__init__(*args, **kwargs)
        self.monitor_thread = hub.spawn(self._monitor)
        self.datapaths = {}
        
        topology_discovery()
    
    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(5)
    
    def _request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        # req = parser.OFPFlowStatsRequest(datapath)
        # datapath.send_msg(req)
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        global byte, clock, bw_used, bw_available, costs

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
                            
                        # Uncomment for DSP
                        # costs[(str(dpid), sw)] = 1/int(bw_available.get((str(dpid), sw), 0))
                        
                        if int(number_flows.get((str(dpid), sw), 0)) == 0 and int(number_flows.get((str(dpid), sw), 0)) == 0:
                            costs[(str(dpid), sw)] = 1/int(bw_available.get((str(dpid), sw), 0))
                        else:
                            costs[(str(dpid), sw)] = (int(number_flows.get((str(dpid), sw), 0))/2 + \
                                int(number_flows.get((str(dpid), sw), 0))/2)/int(bw_available.get((str(dpid), sw), 0))
                            
                    # print(bw_available)
                            
                    byte[(str(dpid), sw)] = stat.tx_bytes
                    clock[(str(dpid), sw)] = time.time()

        for src_host in host_to_switch_port.keys():
            for dst_host in host_to_switch_port.keys():
                if src_host != dst_host:
                    graph = proactive_paths_computation.build_graph_from_txt(costs)
                    paths[(src_host, dst_host)] = proactive_paths_computation.dijkstra_from_macs(
                        graph, src_host, dst_host, host_to_switch_port, adjacency)
                    self.install_path(paths[(src_host, dst_host)], src_host, dst_host)
        
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
                
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()

        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        
        self.add_flow(datapath, 0, match, actions)
        
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
            
        datapath.send_msg(mod)
       
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        global ended_flows, active_paths, last_active_paths_len, requested_flows
        
        body = ev.msg.body
        temp_paths = {}
        
        for item in body:
            eth_src = item.match.get('eth_src')
            eth_dst = item.match.get('eth_dst')
            # filter registered flows by their src and dst, which need to be defined
            if eth_src != None and eth_dst != None:
                # for the pair src, dst, fetch the saved path
                temp_paths[(eth_src, eth_dst)] = paths.get((eth_src, eth_dst))
        
        # if both (src, dst) and (dst, src) were filtered, the flow is running
        active_paths = {k:v for (k,v) in temp_paths.items() if k in temp_paths.keys() and k[::-1] in temp_paths.keys()}
        
        # save the remaining entries, that represent finished flows (iperfs)
        diff = {k : temp_paths[k] for k in set(temp_paths) - set(active_paths)}
        
        # to prevent misreads (len(active_paths) - len(diff))
        # if there are any active flows (len(active_paths) > 0)
        # protection against false reads (len(active_paths) >= last_active_paths_len/2)
        # when all the iperfs started are terminated (requested_flows == len(diff))
        if ((len(active_paths) - len(diff)) != 0 and len(active_paths) > 0 and len(active_paths) >= last_active_paths_len/2) \
            or requested_flows == len(diff):
            
            # everytime the active flows reaches a new maximum
            if len(active_paths)/2 + len(diff) > requested_flows:
                # /2 because both (src, dst) and (dst, src) exist
                requested_flows = int(len(active_paths)/2)
                
            # if there are finished flows, the number of active paths is lower than it was and the filtering was done correctly
            if len(diff) > 0 and len(active_paths) < last_active_paths_len and len(active_paths) % 2 == 0:
                paths.pop(list(diff.keys())[0], None)
                p_dict.pop(list(diff.keys())[0], None)
                paths.pop(list(diff.keys())[0][::-1], None)
                p_dict.pop(list(diff.keys())[0][::-1], None)
            
            # update the number of flows
            self.update_flows()    
            # remember the number of active paths  
            last_active_paths_len = len(active_paths)
            
        # print("ACTIVE PATHS:", active_paths)      
        # print("N FLOWS:", json.dumps(number_flows))  
        
    def update_flows(self):
        global n_flows, active_paths
        
        for k, v in number_flows.items():
            for k1 in v.keys():
                number_flows[str(k)][str(k1)] = 0
        for item in active_paths:
            if active_paths[item] != None:
                for pos in range(len(active_paths[item])-1):
                    number_flows[str(active_paths[item][pos])][str(active_paths[item][pos+1])] = \
                        number_flows[str(active_paths[item][pos])][str(active_paths[item][pos+1])] + 1   
        
    def install_path(self, p, src_mac, dst_mac):
        for sw, in_port, out_port in p:
            datapath = self.datapaths.get(int(sw))
            parser = datapath.ofproto_parser
            match = parser.OFPMatch(in_port=in_port, eth_src=src_mac, eth_dst=dst_mac)
            actions = [parser.OFPActionOutput(out_port)]
            self.add_flow(datapath, 1, match, actions)
             
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        ofp_parser = dp.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        pkt = pkt.get_protocol(arp.arp)
        print(pkt)
        if not pkt: 
            return  
        
        src = pkt.src_mac
        dst = host_ip_mac[pkt.dst_ip]
        dpid = dp.id
        
        out_port = None
        if paths.get((src, dst)) != None:
            path = paths[(src, dst)]
            print(src, dst, path, dpid)
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
