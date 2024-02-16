# RL-Traffic-Control

## Traffic Control at Traffic Light Intersection with Reinforcement Learning
### Falbo Andrea 887525 - Tenderini Ruben 879290

## Table of Contents

### 1. [Objective](#objective)
### 2. [Contents](#contents)
### 3. [Tools](#tools)
### 4. [Codebase](#codebase)
### 5. [Study](#study)
### 6. [References](#references)

## Objective

The objective of this project is to compare different reinforcement learning algorithms in controlling traffic flow at a traffic light intersection. We aim to evaluate their performance under various traffic conditions and configurations.

## Contents

- We utilized the Big Intersection as our traffic intersection model.
- Three traffic configurations were considered: low, medium, and high.
- In addition to Fixed-Time Control, we implemented four reinforcement learning algorithms:
  - Q-Learning
  - Deep Q-Learning
  - Sarsa
  - Sarsa Decay
  
  Sarsa Decay is similar to Sarsa, but it utilizes epsilon greedy to decay epsilon. Notably, it was not implemented in the Sumo-RL repository.

## Tools

The following tools were used in this project:

- Sumo
- Sumo-RL
- Stable Baselines 3
- 2WSI-RL
- Deep Q-Learning agent for Traffic Signal Control
- Matplotlib
- Python
- PyCharm
- Github

## Codebase

- **big-intersection**: This directory houses essential files pertaining to the problem map. It includes:
  - `BI.net.xml`: A detailed configuration file defining the network layout created using SUMO (Simulation of Urban MObility).
  - `BI_50_test.rou.xml`: A route file representing low-traffic scenarios, meticulously designed with SUMO.
  - `BI_150_test.rou.xml`: A route file depicting high-traffic scenarios, crafted using SUMO's capabilities.

- **configs**: Here, you'll find various configuration files tailored for different reinforcement learning algorithms utilized during training sessions:
  - `config_dqn.yaml`: This YAML file encapsulates the configuration parameters for the Deep Q-Network (DQN) algorithm, known for its prowess in handling complex decision-making tasks.
  - `config_ql.yaml`: Within this file lies the setup for Q-Learning (QL), a classic algorithmic approach to reinforcement learning, renowned for its simplicity and effectiveness.
  - `config_fixed.yaml`: This configuration file defines parameters for a fixed cycle strategy, offering a stable baseline for comparison with dynamic algorithms.
  - `config_sarsa.yaml`: SARSA (State-Action-Reward-State-Action) algorithm configuration is stored here, providing a framework for temporal difference learning with on-policy updates.
  - `config_sarsayaml`: SARSA with epsilon-greedy exploration is configured in this file, enabling a balance between exploration and exploitation in the learning process.
  - `test.yaml`: This YAML file consolidates configurations for the final testing phase, ensuring seamless evaluation and benchmarking of the trained models.

- **output**: Within this directory, you'll discover a treasure trove of files generated by the project's scripts:
  - **csv**: A collection of CSV files, each corresponding to a specific reinforcement learning algorithm and phase, containing valuable data for analysis and visualization.
  - **plots**: An assortment of plots and visualizations, offering insights into the performance and behavior of different algorithms across various phases of training and testing.
  - **model**: This directory houses trained models of the algorithms, each representing a culmination of learning and adaptation to the traffic management domain.

- **scripts**: The heart of the project, this directory harbors all Python scripts necessary for execution:
  - **agents**: Here reside Python files embodying the intelligence of various reinforcement learning algorithms:
    - `dqn_agent.py`: Implementation of the Deep Q-Network (DQN) agent, adept at handling complex decision-making tasks through deep neural networks.
    - `ql_agent.py`: Implementation of the Q-Learning (QL) agent, leveraging tabular methods to learn optimal policies in dynamic environments.
    - `fixed_cycle.py`: Implementation of a fixed cycle strategy, providing a stable reference point for evaluating the performance of dynamic algorithms.
    - `learning_agent.py`: An abstract file containing methods and utilities inherited by all other reinforcement learning algorithms, promoting code reusability and maintainability.
    - `sarsa_agent.py`: Implementation of the State-Action-Reward-State-Action (SARSA) algorithm, facilitating temporal difference learning with on-policy updates.
    - `sarsa_agent_decay.py`: Extending SARSA, this file implements epsilon-greedy exploration to balance between exploration and exploitation during learning.
  - **custom**: This directory hosts custom wrapper files tailored for enhanced integration with SUMO-RL:
    - `custom_environment.py`: A wrapper providing enhanced functionality and abstraction for interfacing with the SUMO environment, streamlining interaction and data handling.
    - `custom_true_online_sarsa.py`: A specialized wrapper facilitating the implementation of SARSA with decay, offering improved convergence and stability during training.
  - **utils**: A repository of utility scripts indispensable for the smooth operation of the project:
    - `config_parser.py`: A robust parser for configuration files, enabling seamless extraction and utilization of algorithmic parameters.
    - `config_values.py`: A comprehensive collection of possible values and configurations, ensuring consistency and reliability across different setups.
    - `plotter.py`: An essential tool for generating insightful plots and visualizations, aiding in the analysis and interpretation of experimental results.
    - `runner.py`: A versatile script orchestrating the execution of the project, managing training sessions, testing phases, and result generation with ease and efficiency.
  - **docs**: A repository of documentation and research materials essential for understanding and extending the project:
    - `relazione.pdf`: A detailed report documenting the project's objectives, methodologies, results, and conclusions, presented in Italian.
    - `relation.pdf`: A translated version of the report, catering to an English-speaking audience and facilitating broader dissemination and understanding.

- **main.py**: The central execution file of the project, orchestrating the integration of various components, from data preprocessing and algorithmic training to evaluation and result generation.

## Study

All the theoretical background, study, and experiments conducted are documented in the 'docs' folder in both English (relation.pdf) and Italian (relazione.pdf).

## References

[Placeholder for references]