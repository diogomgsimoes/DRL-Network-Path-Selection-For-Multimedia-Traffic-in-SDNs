from ryu.base import app_manager
from ryu.controller import mac_to_port
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.mac import haddr_to_bin
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib import mac
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link
from ryu.app.wsgi import ControllerBase
from collections import defaultdict
from ryu.lib import hub
from operator import attrgetter
from datetime import datetime
import time
import json

_switches = []
mymac = {}

adjacency = defaultdict(lambda: defaultdict(lambda: None))
datapath_list = {}
byte = defaultdict(lambda: defaultdict(lambda: None))
clock = defaultdict(lambda: defaultdict(lambda: None))
bw_used = defaultdict(lambda: defaultdict(lambda: None))
bw_available = defaultdict(lambda: defaultdict(lambda: None))
bw = defaultdict(lambda: defaultdict(lambda: None))
costs = defaultdict(lambda: defaultdict(lambda: None))
n_flows = defaultdict(lambda: defaultdict(lambda: None))
paths = {}
active_paths = {}
last_active_paths_len = 0
ended_flows = 0
p_dict = {}
requested_flows = 0


def get_path_dynamic(src, dst, first_port, final_port, src_ip, dst_ip):
    # nested dict to add the calculated paths
    global paths
    # save the destination
    dest = dst

    _switches_keys = set(_switches)
    distance = {}
    previous = {}

    # for each switch set max distance and no previous hop
    for dpid in _switches_keys:
        distance[dpid] = float('Inf')
        previous[dpid] = None

    # set src to 0 distance
    distance[src] = 0

    while len(_switches_keys) > 0:
        # closest neighbour
        closest_node = minimum_distance(distance, _switches_keys)
        # choose it by removing from the set
        _switches_keys.remove(closest_node)
        
        for node in _switches_keys:
            # if there is a port connecting the nodes (next neighbour)
            if adjacency[closest_node][node] != None:
                w = costs[str(closest_node)][str(node)]
                # Uncomment for SP
                # w = 1
                if distance[closest_node] + w < distance[node]:
                    distance[node] = distance[closest_node] + w
                    previous[node] = closest_node

    path = []
    path.append(dest)
    current_node = previous[dest]

    # iterate through the previous dict until we reach the src
    while current_node is not None:
        if current_node == src:
            path.append(current_node)
            break
        
        dest = current_node
        path.append(dest)
        current_node = previous[dest]

    path.reverse()

    if src == dest:
        path = [src]
    
    # add to the paths dict
    paths[(src_ip,dst_ip)] = path
    paths[(dst_ip,src_ip)] = path
   
    # add the ports
    ports = []
    in_port = first_port

    # zip function for parallel iteration
    for s1, s2 in zip(path[:-1], path[1:]):
        out_port = adjacency[s1][s2]
        ports.append((s1, in_port, out_port, distance[s2]))
        in_port = adjacency[s2][s1]

    ports.append((dst, in_port, final_port, distance[dst]))
    return ports

def minimum_distance(distance, Q):
    min = float('Inf')
    node = 0

    for v in Q:
        if distance[v] < min:
            min = distance[v]
            node = v
    return node

class ProjectController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ProjectController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.topology_api_app = self
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        global bw, costs, n_flows

        # initialize the nested dicts with the switches and links data
        try:
            fin = open("topology_info.txt", "r")
            for line in fin:
                a = line.split()
                if a:
                    bw[str(a[0])][str(a[1])] = int(a[2])
                    bw[str(a[1])][str(a[0])] = int(a[2])
                    costs[str(a[0])][str(a[1])] = 0
                    costs[str(a[1])][str(a[0])] = 0
                    n_flows[str(a[0])][str(a[1])] = 0
                    n_flows[str(a[1])][str(a[0])] = 0  

            fin.close()

        except IOError:
            print("make topology_info.txt not ready")

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath

        if ev.state == MAIN_DISPATCHER:
            if not datapath.id in self.datapaths:
                print('register datapath:', datapath.id)
                self.datapaths[datapath.id] = datapath

        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                print('unregister datapath:', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(3)
    
    def _request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

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
            
        print("ACTIVE PATHS:", active_paths)      
        print("N FLOWS:", json.dumps(n_flows))  
            
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        global byte, clock, bw_used, bw_available, costs

        body = ev.msg.body
        dpid = ev.msg.datapath.id

        for stat in sorted(body, key=attrgetter('port_no')):
            for sw in _switches:
                if adjacency[dpid][sw] == stat.port_no:
                    if byte[dpid][sw] != None:
                        if byte[dpid][sw] > 0:
                            bw_used[dpid][sw] = (stat.tx_bytes - byte[dpid][sw]) * 8.0 \
                                / (time.time() - clock[dpid][sw]) / 1000

                            bw_available[str(dpid)][str(sw)] = int(bw[str(dpid)][str(sw)]) \
                                * 1024.0 - bw_used[dpid][sw]
                            
                            # Uncomment for DSP
                            # costs[str(dpid)][str(p)] = 1/bw_available[str(dpid)][str(p)]
                            
                            if n_flows[str(dpid)][str(sw)] == 0 and n_flows[str(sw)][str(dpid)] == 0:
                                costs[str(dpid)][str(sw)] = 1/bw_available[str(dpid)][str(sw)]
                            else:
                                costs[str(dpid)][str(sw)] = (n_flows[str(dpid)][str(sw)]/2 + \
                                    n_flows[str(sw)][str(dpid)]/2)/bw_available[str(dpid)][str(sw)]
                            
                            print("BANDWIDTH: ", json.dumps(bw_available))
                            
                    byte[dpid][sw] = stat.tx_bytes
                    clock[dpid][sw] = time.time()
                    
    def update_flows(self):
        global n_flows, active_paths
        
        for k, v in n_flows.items():
            for k1 in v.keys():
                n_flows[str(k)][str(k1)] = 0
        for item in active_paths:
            if active_paths[item] != None:
                for pos in range(len(active_paths[item])-1):
                    n_flows[str(active_paths[item][pos])][str(active_paths[item][pos+1])] = \
                        n_flows[str(active_paths[item][pos])][str(active_paths[item][pos+1])] + 1   

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match, idle_timeout=3,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority, idle_timeout=3,
                                    match=match, instructions=inst)
            
        datapath.send_msg(mod)

    def install_path(self, p, ev, src_mac, dst_mac):
        msg = ev.msg
        datapath = msg.datapath
        parser = datapath.ofproto_parser

        for sw, in_port, out_port, cost in p:
            match = parser.OFPMatch(in_port=in_port, eth_src=src_mac, eth_dst=dst_mac)
            actions = [parser.OFPActionOutput(out_port)]
            datapath = datapath_list[sw]
            self.add_flow(datapath, 1, match, actions)

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

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        
        # initialize mac_to_port for the switch
        self.mac_to_port.setdefault(dpid, {})
        
        # save the mac address to avoid flooding (added)
        self.mac_to_port[dpid][src] = in_port

        if src not in mymac.keys():
            mymac[src] = (dpid, in_port)
        
        # mymac.keys()
        if dst in self.mac_to_port[dpid]:
            if p_dict.get((src, dst)) != None:
                p = p_dict.get((src, dst))
            else:
                p = get_path_dynamic(mymac[src][0], mymac[dst][0], mymac[src][1], mymac[dst][1], src, dst)
                p_dict[(src, dst)] = p
            self.install_path(p, ev, src, dst)
            out_port = self.mac_to_port[dpid][dst]
            # out_port = p[0][2]
        else:
            # if the dst mac address wasn't received earlier
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]
        
        # if out_port != ofproto.OFPP_FLOOD:
        #     match = parser.OFPMatch(in_port=in_port, eth_src=src, eth_dst=dst)
            
        data = None
        
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
            
        # send packetOut message
        if out_port == ofproto.OFPP_FLOOD:
            while len(actions) > 0:
                actions.pop()
                
            for i in range(1, 23):
                actions.append(parser.OFPActionOutput(i))
                
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
									in_port=in_port, actions=actions, data=data)
            datapath.send_msg(out)   
        else:
            out = parser.OFPPacketOut(
				datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port,
				actions=actions, data=data)
            datapath.send_msg(out)
            
    events = [
		event.EventSwitchEnter,
		event.EventSwitchLeave,
		event.EventPortAdd,
		event.EventPortDelete,
		event.EventPortModify,
		event.EventLinkAdd,
		event.EventLinkDelete]

    @set_ev_cls(events)
    def get_topology_data(self, ev):
        global _switches, adjacency, datapath_list

        # get all switches
        switch_list = get_switch(self.topology_api_app, None)
        
        # get switches id's
        _switches = [switch.dp.id for switch in switch_list]
        
        # setting datapath_dict to {sw_id: sw_datapath}
        datapath_list = dict([(switch.dp.id, switch.dp) for switch in switch_list])

        # get all links
        links_list = get_link(self.topology_api_app, None)
        
        # create list of connections
        _links = [(link.src.dpid, link.dst.dpid, link.src.port_no, link.dst.port_no) for link in links_list]

        for s1, s2, port1, port2 in _links:
            adjacency[s1][s2] = port1
            adjacency[s2][s1] = port2
