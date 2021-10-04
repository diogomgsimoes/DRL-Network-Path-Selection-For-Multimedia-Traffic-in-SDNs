# DRL-based Network Path Selection For Multimedia Traffic in SDNs #

This repository fullfils its supporting role to a master thesis on "Intelligent Routing for Software-Defined Media Networks". It explores two ways to optimize routing in the transport of multimedia content in SDNs. This project was developed in Python, inside an Ubuntu VMWare virtual machine and relies on three main Python packages: OpenAI Gym, PyTorch and NetworkX, as well as Mininet.

These approaches at routing optimization explored in this project are divided into two folders:
  - link_cost_routing_optimization,
  - drl_routing_optimization.

In the main folder, scripts that are required in both approaches can be found. Among these general files there is a topology text file, which can altered. However, if so, there are numerous parameters in other scripts than need to be changed accordingly.

To simulate the link cost optimization algorithms, the user must run the Ryu controller script with the desired cost equation uncommented (there are three available) and the simulating script.

Terminal 1:
```
ryu-manager --observe-links proactive_ryu_controler.py
```
Terminal 2:
```
sudo python3.8 -E proactive_baseline_performance_tester.py
```

On the other hand, to simulate DRL agents, the DRL version of the Ryu controller must be launched as well as the model evaluation script filled with the name of the desired model's parameters.

Terminal 1:
```
ryu-manager --observe-links proactive_drl_controler.py
```
Terminal 2:
```
sudo python3.8 -E model_evaluation.py
```

To train the different agents in each setup, the respetive file must be ran using:
```
sudo python3.8 -E <agent_filename>.py
```
