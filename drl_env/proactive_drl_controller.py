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
import socket

import proactive_paths_computation

TOPOLOGY_FILE_NAME = 'topology.txt'
NUMBER_SWITCHES = 10

host_to_switch_port = defaultdict(lambda: defaultdict(lambda: None))
adjacency = {}
paths = {}
active_paths = {}
byte = {}
clock = {}
bw_used = {}
bw_available = {}
bw = {}
switch_ports = {}
_switches = []
host_ip_mac = {}
paths_hops = {}


def topology_discovery():
    global adjacency, host_to_switch_port, switch_ports, _switches, host_ip_mac
    
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
                _switches.append(sw1) if sw1 not in _switches else None
                _switches.append(sw2) if sw2 not in _switches else None

def load_paths():
    global active_paths
    
    try:
        f = open("active_paths.txt", "r")
        for line in f:
            a = line.strip('\n').split('_')
            if a:
                path_list = list(a[2].split(","))
                for i in range(len(path_list)):
                    if "[" in path_list[i]:
                        path_list[i] = path_list[i].replace("[", "")
                        path_list[i] = path_list[i].replace(" ", "")
                        path_list[i] = path_list[i].replace("'", "")
                    elif "]" in path_list[i]:
                        path_list[i] = path_list[i].replace("]", "")
                        path_list[i] = path_list[i].replace(" ", "")
                        path_list[i] = path_list[i].replace("'", "")
                    else:
                        path_list[i] = "S" + path_list[i].replace(" ", "")
                        
                src = a[0].replace("H", "")
                src_mac = "00:00:00:00:00:{}".format(src.zfill(2))
                dst = a[1].replace("H", "")
                dst_mac = "00:00:00:00:00:{}".format(dst.zfill(2))

                active_paths[(src_mac, dst_mac)] = path_list
                
        f.close()

    except IOError:
        print("file not ready")

                    
class DRLProactiveController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DRLProactiveController, self).__init__(*args, **kwargs)
        self.monitor_thread = hub.spawn(self._monitor)
        self.datapaths = {}
        
        topology_discovery()

    def _monitor(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', 6631))
        s.listen()
        conn, _ = s.accept()
        
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(3)
            
            _str = ""
            for item in bw_available:
                _str = _str + item[0] + "_" + item[1] + "_" + str(bw_available[item]) + "/"
                
            conn.sendall(len(_str).to_bytes(4, 'little'))
            conn.sendall(_str.encode('utf-8'))
        
    def _request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        # req = parser.OFPFlowStatsRequest(datapath)
        # datapath.send_msg(req)
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        pass 
            
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        global byte, clock, bw_used, bw_available

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
                        
                    # print(bw_available)
                        
                    byte[(str(dpid), sw)] = stat.tx_bytes
                    clock[(str(dpid), sw)] = time.time()
                    
        if len(self.datapaths) == NUMBER_SWITCHES:
            load_paths()
            self.update_paths()   
                
    def update_paths(self):
        global paths_hops, paths
        
        for path in active_paths.values():
            src_mac = "00:00:00:00:00:{}".format(path[0][1:].zfill(2))
            dst_mac = "00:00:00:00:00:{}".format(path[-1][1:].zfill(2))
            saved_path = paths_hops.get((src_mac, dst_mac))
            if saved_path != path:
                self.uninstall_path(paths[(src_mac, dst_mac)], src_mac, dst_mac)
                paths_hops[(src_mac, dst_mac)] = path
                paths[(src_mac, dst_mac)] = proactive_paths_computation.add_ports_to_path(path, host_to_switch_port, adjacency, src_mac, dst_mac)
                self.install_path(paths[(src_mac, dst_mac)], src_mac, dst_mac)

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
            self.install_starting_rules()     
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        
        # installing a table-miss flow entry in the switch

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
        
    def remove_flow(self, datapath, match):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        mod = parser.OFPFlowMod(datapath=datapath, match=match, priority=1,
                                command=ofproto.OFPFC_DELETE, out_group=ofproto.OFPG_ANY, out_port=ofproto.OFPP_ANY)
            
        datapath.send_msg(mod)
        
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

    def install_starting_rules(self):
        global paths, paths_hops
        
        for src_host in host_to_switch_port.keys():
                for dst_host in host_to_switch_port.keys():
                    if src_host != dst_host:
                        graph = proactive_paths_computation.build_graph_from_txt()
                        path = proactive_paths_computation.dijkstra_from_macs(
                            graph, src_host, dst_host, host_to_switch_port, adjacency)
                        paths_hops[(src_host, dst_host)] = path
                        paths[(src_host, dst_host)] = proactive_paths_computation.add_ports_to_path(
                            path, host_to_switch_port, adjacency, src_host, dst_host)
                        self.install_path(paths[(src_host, dst_host)], src_host, dst_host)
                    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
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
