# DRL-based Network Path Selection For Multimedia Traffic in SDNs #

This repository fullfils its supporting role to a master thesis on "Intelligent Routing for Software-Defined Media Networks". It explores two ways to optimize routing in the transport of multimedia content in SDNs. This project was developed in Python, inside an Ubuntu VMWare virtual machine and relies on three main Python packages: OpenAI Gym, PyTorch and NetworkX, as well as Mininet.

These approaches at routing optimization explored in this project are divided into two folders:
  - link_cost_routing_optimization,
  - drl_routing_optimization.

In the main folder, scripts that are required in both approaches can be found. Among these general files there is a topology text file, which can altered but needs to follow the same structure of the current one to assure that the remaining scripts will work. However, if so, there are numerous parameters in other scripts than need to be changed accordingly, such as:
 - "proactive_paths_computation.py": change the topology filename. This script contains helper functions that take advantage of NetworkX.
 - "proactive_topology_mininet.py": change the topology filename. This script builds a Mininet network from a text file.

In regards to the approach-specific scripts, there are also a few parameters that need to be tuned when using different topologies:
 - "proactive_ryu_controller.py" or "proactive_drl_controller.py": change the number of switches and topology filename.
 - "mininet_env.py": change the number of hosts and desired paths between each hosts pair.

To simulate the link cost optimization algorithms, the user must run the Ryu controller script with the desired cost equation uncommented (there are three available) and the simulating script.

Terminal 1:
```
ryu-manager --observe-links proactive_ryu_controller.py
```
Terminal 2:
```
sudo python3.8 -E proactive_baseline_performance_tester.py
```

On the other hand, to simulate DRL agents, the DRL version of the Ryu controller must be launched as well as the model evaluation script filled with the name of the desired model's parameters.

Terminal 1:
```
ryu-manager --observe-links proactive_drl_controller.py
```
Terminal 2:
```
sudo python3.8 -E model_evaluation.py
```

In this second script, the file must be adapted to the agent being simulated (DQN, DDQN or Dueling DQN). The respective neural network constructor and parameters must be defined according to the model parameters file that will be tested. 

The current iPerf simulation settings are:
 - 15 Mbits/s bitrate;
 - TCP;
 - JSON output file;
 - 180 seconds duration;
 - Interval of 5 seconds between requests;
 - 32 concurrent requests.

To train the different agents in each setup and create a model to use in the model evaluation script, the respetive file must be ran using:
```
sudo python3.8 -E <agent_filename>.py
```

On both the training (drl version) and testing phases (both versions), the number of requests and their edge nodes can define the results obtained. In this work, an algorithm to select hosts that cause network congestion faster was used and is available by running:
```
sudo python3.8 -E max_centrality_host_pairs.py
```

The resulting output is used in some way in files such as "mininet_env.py", "proactive_mininet_api.py" and "proactive_baseline_performance_tester.py".

To analyze the simulations' results, an helper script can be used, which reads the iPerf report json file and outputs only the bitrate and RTT metrics we were interested in.
```
sudo python3.8 -E read_iperfs.py
```

Lastly, the folder "agents" and the notebook "DRL_routing" were used during the development phase, but not add anything to the solution presented. They were only kept in this repository as a past work record.
