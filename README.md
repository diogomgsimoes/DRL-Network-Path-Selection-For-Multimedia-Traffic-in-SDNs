# RYU-Controller-DRL_vs_Dijkstra-Mininet #

## DRL version: ##
Scripts: proactive_drl_controller + proactive_topology_mininet + proactive_paths_computation + proactive_mininet_api + mininet_env + (env_test if testing or dqn_tf_agent if training)

### TODO: ###
1) Optimize reward model (consider other metrics as well)
2) Optimize agent parameters
3) Build a more robust agent (?)


## Baseline version: ## 
Scripts: proactive_topology_mininet + proactive_ryu_controller

### TODO: ###
1) Add flow number control stat
2) Stop path changes midst iperf
3) Develop simulation script