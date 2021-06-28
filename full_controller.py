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

# adjacency map [sw1][sw2]->port from sw1 to sw2
adjacency = defaultdict(lambda: defaultdict(lambda: None))
datapath_list = {}
byte = defaultdict(lambda: defaultdict(lambda: None))
clock = defaultdict(lambda: defaultdict(lambda: None))
bw_used = defaultdict(lambda: defaultdict(lambda: None))
bw_available = defaultdict(lambda: defaultdict(lambda: None))
bw = defaultdict(lambda: defaultdict(lambda: None))
costs = defaultdict(lambda: defaultdict(lambda: None))
n_flows = defaultdict(lambda: defaultdict(lambda: None))

target_srcmac = "00:00:00:00:00:01"
target_dstmac = "00:00:00:00:00:03"


def get_path_dynamic(src, dst, first_port, final_port):
    global bw_available
    dest = dst
    _switches_keys = set(_switches)
    
    print("Searching for path between %s and %s" % (src, dst))
    
    distance = {}
    previous = {}

    for dpid in _switches_keys:
        distance[dpid] = float('Inf')
        previous[dpid] = None

    distance[src] = 0

    while len(_switches_keys) > 0:
        closest_node = minimum_distance(distance, _switches_keys)
        _switches_keys.remove(closest_node)
        
        for node in _switches_keys:
            if adjacency[closest_node][node] != None:
                #w = costs[str(closest_node)][str(node)]
                w = 1
                if distance[closest_node] + w < distance[node]:
                    distance[node] = distance[closest_node] + w
                    previous[node] = closest_node

    path = []
    path.append(dest)
    current_node = previous[dest]

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
        distance[src] = adjacency[src][dest]

    print(json.dumps(costs))
    print("Shortest path between ", src, " and ", dst, " is: ",
        path, " and the total distance is: ", distance[dst])
   
    # Now add the ports
    r = []
    in_port = first_port

    for s1, s2 in zip(path[:-1], path[1:]):
        out_port = adjacency[s1][s2]
        r.append((s1, in_port, out_port, distance[s2]))
        in_port = adjacency[s2][s1]
        n_flows[str(s1)][str(s2)] = n_flows[str(s1)][str(s2)] + 1

    r.append((dst, in_port, final_port, distance[dst]))
    return r


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
        global bw
        global costs
        global n_flows

        try:
            fin = open("topology_info.txt", "r")
            for line in fin:
                a = line.split()
                if a:
                    bw[str(a[0])][str(a[1])] = int(a[2])
                    bw[str(a[1])][str(a[0])] = int(a[2])
                    costs[str(a[0])][str(a[1])] = 0
                    costs[str(a[1])][str(a[0])] = 0
                    n_flows[str(a[0])][str(a[1])] = int(a[4])
                    n_flows[str(a[1])][str(a[0])] = int(a[4])      

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
        body = ev.msg.body
        #print("----------------\nFLOW ADDED: ", body)
        #print("\n", len(body), "\n")

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        global byte, clock, bw_used, bw_available, costs

        body = ev.msg.body
        dpid = ev.msg.datapath.id

        for stat in sorted(body, key=attrgetter('port_no')):
            for p in _switches:
                if adjacency[dpid][p] == stat.port_no:
                    if byte[dpid][p] != None:
                        if byte[dpid][p] > 0:
                            bw_used[dpid][p] = (stat.tx_bytes - byte[dpid][p]) * \
                                8.0 / (time.time()-clock[dpid][p]) / 1000

                            bw_available[str(dpid)][str(p)] = int(
                                bw[str(dpid)][str(p)]) * 1024.0 - bw_used[dpid][p]
                            
                            costs[str(dpid)][str(p)] = 1/bw_available[str(dpid)][str(p)]
                            
                            print("WEIGHTS UPDATED ", json.dumps(bw_available))

                    byte[dpid][p] = stat.tx_bytes
                    clock[dpid][p] = time.time()
                    #print("WEIGHTS UPDATED ", clock[dpid][p])
            
        # print("--------------------")        
        # print(json.dumps(bw))
        # print("--------------------")
        # print(json.dumps(bw_available))
        # print("--------------------")    
        # print(json.dumps(costs))
        # print("--------------------")    


    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        #match = datapath.ofproto_parser.OFPMatch(in_port=in_port, eth_dst=dst)

        # inst = [parser.OFPInstructionActions(
        #     ofproto.OFPIT_APPLY_ACTIONS, actions)]

        # mod = datapath.ofproto_parser.OFPFlowMod(
        #     datapath=datapath, match=match, cookie=0,
        #     command=ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0,
        #     priority=ofproto.OFP_DEFAULT_PRIORITY, instructions=inst)

        # datapath.send_msg(mod)
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
    
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        # Once the message is created,
        # Ryu will take care of the rest to ensure the message is properly encoded and sent to the switch.
        datapath.send_msg(mod)

    def install_path(self, p, ev, src_mac, dst_mac):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        for sw, in_port, out_port, cost in p:
            #print(src_mac, "->", dst_mac, "via ", sw,
            #     " in_port=", in_port, " out_port=", out_port, " cost=", cost)

            match = parser.OFPMatch(
                in_port=in_port, eth_src=src_mac, eth_dst=dst_mac)
            
            actions = [parser.OFPActionOutput(out_port)]
            
            datapath = datapath_list[sw]
            
            # inst = [parser.OFPInstructionActions(
            #     ofproto.OFPIT_APPLY_ACTIONS, actions)]
            
            # mod = datapath.ofproto_parser.OFPFlowMod(
            #     datapath=datapath, match=match, idle_timeout=0, hard_timeout=0,
            #     priority=1, instructions=inst)

            # datapath.send_msg(mod)
            self.add_flow(datapath, 1, match, actions)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()

        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]

        # inst = [parser.OFPInstructionActions(
        #     ofproto.OFPIT_APPLY_ACTIONS, actions)]

        # mod = datapath.ofproto_parser.OFPFlowMod(
        #     datapath=datapath, match=match, cookie=0,
        #     command=ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0,
        #     priority=0, instructions=inst)

        # datapath.send_msg(mod)
        
        self.add_flow(datapath, 0, match, actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        global target_srcmac, target_dstmac

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
        
        #dpid = format(datapath.id, "d").zfill(16)
        
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})

        if src not in mymac.keys():
            mymac[src] = (dpid, in_port)
            
        #self.mac_to_port[dpid][src] = in_port
            
        if dst in mymac.keys():
            #if (src == target_srcmac and dst == target_dstmac) or (dst == target_srcmac and src == target_dstmac):
            p = get_path_dynamic(mymac[src][0], mymac[dst][0],
                        mymac[src][1], mymac[dst][1])

            self.install_path(p, ev, src, dst)
            out_port = p[0][2]
            # else:
            #     out_port = ofproto.OFPP_FLOOD
            
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]
        
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(
				in_port=in_port, eth_src=src, eth_dst=dst)
            
        data = None
        
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
            
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

        switch_list = get_switch(self.topology_api_app, None)
        _switches = [switch.dp.id for switch in switch_list]

        for switch in switch_list:
            datapath_list[switch.dp.id] = switch.dp

       # print("Current switches = ", _switches)

        links_list = get_link(self.topology_api_app, None)
        mylinks = [(link.src.dpid, link.dst.dpid, link.src.port_no,
                    link.dst.port_no) for link in links_list]

        for s1, s2, port1, port2 in mylinks:
            adjacency[s1][s2] = port1
            adjacency[s2][s1] = port2
          # print("Connection: sw ", s1, ", port ", port1, " <---> sw", s2, ", port", port2, ", cost", costs[str(s1)][str(s2)])