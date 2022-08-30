# DRL-based Network Path Selection For Multimedia Traffic in SDNs #

This repository fullfils its supporting role to a master thesis on "Intelligent Routing for Software-Defined Media Networks", summarized in the following paper https://ieeexplore.ieee.org/document/9793657. It explores two ways of optimizing routing in the transport of multimedia content in SDNs. This project was developed in Python, inside an Ubuntu VMWare virtual machine and relies on three main Python packages: OpenAI Gym, PyTorch and NetworkX, as well as Mininet.

The routing optimization approaches explored in this project are divided into two folders:
  - link_cost_routing_optimization,
  - drl_routing_optimization.

In the root folder, scripts that are required in both approaches can be found. Among these general files there is a topology text file, which can altered but needs to maintain its structure to assure that the remaining scripts will work. Additionally, if changed, there are numerous parameters in other scripts than need to be altered accordingly, such as:
 - "proactive_paths_computation.py": change the topology filename. This script contains helper functions that take advantage of NetworkX.
 - "proactive_topology_mininet.py": change the topology filename. This script builds a Mininet network from a text file.

In regards to the approach-specific scripts, there are also a few parameters that need to be tuned when using different topologies:
 - "proactive_ryu_controller.py" or "proactive_drl_controller.py": change the number of switches and topology filename.
 - "mininet_env.py": change the number of hosts and desired paths between each hosts pair.

To simulate the link cost optimization algorithms, the user must run the Ryu controller script with the desired cost equation uncommented (there are three available in "proactive_ryu_controller.py") and the simulating script.

Terminal 1:
```
ryu-manager --observe-links proactive_ryu_controller.py
```
Terminal 2:
```
sudo python3.8 -E proactive_baseline_performance_tester.py
```

On the other hand, to simulate the DRL agents, the DRL version of the Ryu controller must be launched as well as the model evaluation script filled with the name of the desired model's parameters file.

Terminal 1:

```
ryu-manager --observe-links proactive_drl_controller.py
```

Terminal 2:

```
sudo python3.8 -E model_evaluation.py
```

In this second script, the file must be adapted to the agent being simulated (DQN, DDQN or Dueling DQN). The respective neural network constructor and parameters must be defined according to the model parameters file that will be tested (comments on the file explain how). 

The current iPerf simulation settings are:
 - 15 Mbits/s bitrate;
 - TCP;
 - JSON output file;
 - 180 seconds duration;
 - Interval of 5 seconds between requests;
 - 32 concurrent requests.

To train the different agents in each network use setup (the four possible setups are detailed in the thesis) and create a model to use in the model evaluation script, the respetive file must be ran using (alternatively you can train it on your main OS for a faster process):

```
sudo python3.8 -E <agent_filename>.py
```

On both the training (drl version) and testing phases (both versions), the number of requests and their edge nodes have a great influence in the results obtained. In this work, an algorithm to select hosts that cause network congestion faster was used and is available by running:

```
sudo python3.8 -E max_centrality_host_pairs.py
```

The resulting output is used in some way in files such as "mininet_env.py" (number of requests), "proactive_mininet_api.py" and "proactive_baseline_performance_tester.py" (desired host pairs).

To analyze the simulations' results, an helper script can be used, which reads the iPerf report json file and outputs only the bitrate and RTT metrics we were interested in.

```
sudo python3.8 -E read_iperfs.py
```

Lastly, the folder "agents" and the notebook "DRL_routing" were used during the development phase, but do not add anything to the solution presented. They were only kept in this repository as a past work record.
