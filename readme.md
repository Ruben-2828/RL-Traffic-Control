# RL-Traffic-Control

## Traffic Control at Traffic Light Intersection with Reinforcement Learning

#### Tenderini Ruben - 879290
#### Falbo Andrea - 887525

## Table of Contents

### 1. [Objective](#objective)
### 2. [Contents](#contents)
### 3. [Tools](#tools)
### 4. [Setup](#setup)
### 5. [Codebase](#codebase)
### 6. [Study](#study)
### 7  [Collaborate](#collaborate)
### 7. [References](#references)

## Objective

The primary aim of this research is to evaluate various **reinforcement learning algorithms** at a traffic signal junction 
for traffic control. Specifically, we aim to identify the optimal hyperparameters for each algorithm to determine if 
any of them enhances traffic flow compared to traditional **fixed cycle traffic lights**.

## Contents

In our study, we employed the **Big Intersection** as the model for our traffic intersection analysis. 
We aimed to understand how different traffic management strategies perform under two traffic conditions: 
low and high traffic volumes.

Our approach included comparing a traditional traffic management system, the **Fixed-Time Control**, 
with four reinforcement learning algorithms:

1. **Q-Learning**: This technique learns the optimal action to take in various traffic states through trial and error, 
improving over time based on the rewards received for actions taken.

2. **Deep Q-Learning**: An advancement of Q-Learning, this method uses deep neural networks to deal with the complex, 
high-dimensional environments typical of traffic systems, enabling the algorithm to make more nuanced decisions.

3. **Sarsa**: Similar to Q-Learning, Sarsa learns from the current state and action but also considers the next state 
and action, making it slightly more conservative but often more stable in its learning process.

4. **Sarsa Decay**: A variant of Sarsa, this algorithm introduces a decaying epsilon-greedy strategy, 
gradually reducing the exploration of new actions over time to fine-tune its policy.

Noteworthy is that **Sarsa Decay**, unlike the other algorithms, was not pre-implemented in the **Sumo-RL** repository.

## Tools

The main tools and methodologies used to conduct this study are provided below:

### Sumo

[**SUMO**](https://sumo.dlr.de/docs/), acronym for "Simulation of Urban MObility", represents an open-source software dedicated to
detailed simulation of urban mobility. Developed at the Center's Institute of Transportation Systems
German Aerospace (DLR), **SUMO** offers a flexible and highly portable platform.

### Sumo-RL

[**Sumo-RL**](https://github.com/LucasAlegre/sumo-rl) is a repository developed by Lucas N. Alegre, focused on integrating 
**SUMO** with learning algorithms for
reinforcement (**RL**). Its goal is to extend the capabilities of **SUMO**, allowing users to explore and
develop advanced traffic control strategies using **RL** approaches.

### Stable Baselines 3

[**Stable Baselines 3**](https://github.com/DLR-RM/stable-baselines3) is a reinforcement learning (**RL**) library in Python, 
developed by OpenAI.
It provides a set of stable and reliable **RL** algorithm implementations, designed to be easily
accessible and usable by developers. In this project he was also fundamental for
the integration of **Gymnasium** environments.

### 2WSI-RL

[**2WSI-RL**](https://github.com/rChimisso/2WSI-RL), an acronym for 2 Way Single Intersection for Reinforcement Learning, 
is a study on the application of reinforcement learning for the management of a traffic light intersection, specifically 
the 2 Way Single Intersection, hosted by Riccardo Chimisso.

### Deep Q-Learning Agent for Traffic Signal Control

[**Deep Q-Learning Agent for Traffic Signal Control**](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control) 
is a framework where an agent that learns by reinforcement via Q-Learning tries to choose the green phase of the 
intersection to maximize efficiency

### Matplotlib

[**Matplotlib**](https://matplotlib.org/) is a data visualization library in Python, designed to create static, interactive plots
and animations.

### Python

**[Python](https://www.python.org/)** is a high-level programming language widely used in the field of learning
automatic, data processing and in many scientific fields. 

### PyCharm

[**PyCharm**](https://www.jetbrains.com/pycharm/) is an integrated development environment specifically designed for 
the programming language Python.

### Github

[**GitHub**](https://github.com/) is a hosting platform for software projects that uses version control
ment Git. It provides a collaborative environment for software development, allowing developers to
upload, share and manage versioning of their projects.

## Setup

Here's the setup:

1. **Install Python:** You can download and install Python from the official website: [Python.org](https://www.python.org/).

2. **Install SUMO:** You can install SUMO following the instructions on the official website: 
[SUMO Installation](https://sumo.dlr.de/docs/Installing.html)

3. **Set SUMO_HOME:** Set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)

4. **Install an IDE:** You can choose between PyCharm, VSCode, or any other IDE you prefer. You can download 
PyCharm from [JetBrains website](https://www.jetbrains.com/pycharm/) or 
VSCode from [Visual Studio Code website](https://code.visualstudio.com/).

5. **Install SUMO-RL:** You can install SUMO-RL using pip:
   ```
   pip install sumo-rl
   ```

6. **Install Matplotlib:** You can install Matplotlib using pip:
   ```
   pip install matplotlib
   ```

7. **Install Stable Baselines 3:** You can install Stable Baselines 3 using pip:
   ```
   pip install stable-baselines3
   ```

8. **Install pandas:** You can install Pandas using pip:
   ```
   pip install pandas 
   ```
9. **Install os:** You can install OS using pip:
    ```
   pip install os 
   ```
10. **Install pickle :** You can install Pickle using pip:
     ```
    pip install pickle 
    ```
11. **Install yaml:** You can install Yaml using pip:
   ```
   pip install PyYAML 
   ```
12. **Install abc:** You can install Abc using pip:
   ```
   pip install abc 
   ```
13. **Install other packages:** Use pip to install additional Python packages required for your project.
   ```
   pip install *other_packages* 
   ```
14. **Install linear-rl** You can install the linear-rl repository by Lucas Alegre needed for Sarsa using pip:
    ```
    pip install  git+https://github.com/LucasAlegre/linear-rl
    ```

If you're using PyCharm, after following these simple steps, everything should be ready to go! 

The same should apply for VSCode!


## Codebase

- **big-intersection**: This folder holds essential files for our traffic intersection model. It includes:
  - `BI.net.xml`: This file defines how the roads and intersections are laid out using SUMO.
  - `BI_50_test.rou.xml`: It represents scenarios with low traffic, carefully designed using SUMO.
  - `BI_150_test.rou.xml`: It represents scenarios with high traffic, carefully designed using SUMO.

- **configs**: Here, you'll find configuration files for different reinforcement learning algorithms used during 
training sessions, all made in YAML format:
  - `config_dqn.yaml`:  Contains configurations for the Deep Q-Network (DQN) algorithm, known for handling complex tasks
  - `config_ql.yaml`: Holds configurations for Q-Learning (QL), a straightforward but effective algorithmic approach. 
  - `config_fixed.yaml`: This file defines parameters for a fixed cycle strategy, used as a stable reference for 
  comparison. 
  - `config_sarsa.yaml`: Stores configurations for the SARSA algorithm, which focuses on temporal difference 
  learning with on-policy updates. 
  - `config_sarsa_decay.yaml`: Contains configurations for SARSA with epsilon-greedy exploration, 
  balancing exploration and exploitation. 
  - `test.yaml`: Consolidates configurations for the final testing phase, ensuring smooth evaluation of
  trained models.

- **docs**: A repository of documentation and research materials essential for understanding and extending the 
  project:
    - `relazione.pdf`: A detailed report documenting the project's objectives, methodologies, results, and conclusions, 
    presented in Italian.
    - `report.pdf`: A translated version of the report, catering to an English-speaking audience

- **output**: This directory contains various files generated by project scripts:
  - **csv**: Holds CSV files, each corresponding to a specific reinforcement learning algorithm and phase, for analysis.
  - **plots**: Contains visualizations offering insights into algorithm performance during training and testing.
  - **model**: Stores trained models of the algorithms, representing their learned behaviors in the traffic 
  management domain.

- **scripts**: The heart of the project, this directory harbors all Python scripts necessary for execution:
  - **agents**: Includes Python files that define how different reinforcement learning algorithms behave:
    - `dqn_agent.py`: Implementation of the Deep Q-Network (DQN) agent, adept at handling complex decision-making tasks 
    through deep neural networks.
    - `ql_agent.py`: Implementation of the Q-Learning (QL) agent, leveraging tabular methods to learn optimal policies
    in dynamic environments.
    - `fixed_cycle.py`: Implementation of a fixed cycle strategy, providing a stable reference point for evaluating the 
    performance of dynamic algorithms.
    - `learning_agent.py`: A file containing abstracts methods and utilities inherited by all other
    reinforcement learning algorithms
    - `sarsa_agent.py`: Implementation of the State-Action-Reward-State-Action (SARSA) algorithm, facilitating 
    temporal difference learning with on-policy updates.
    - `sarsa_agent_decay.py`: Extending SARSA, this file implements epsilon-greedy exploration to balance between 
    exploration and exploitation during learning.
  - **custom**: Holds special wrapper files customized to work better with SUMO-RL integration.
    - `custom_environment.py`: A wrapper providing enhanced functionality and abstraction for interfacing with 
    the SUMO environment
    - `custom_true_online_sarsa.py`: A specialized wrapper facilitating the implementation of SARSA with decay.
  - **utils**: Contains essential utility scripts that ensure the project runs without any hitches:
    - `config_parser.py`: A robust parser for configuration files, enabling seamless extraction and utilization of 
    algorithmic parameters.
    - `config_values.py`: A comprehensive collection of possible values and configurations, 
    ensuring consistency and reliability across different setups.
    - `plotter.py`: An essential tool for generating insightful plots and visualizations, 
    aiding in the analysis and interpretation of experimental results.
    - `runner.py`: A script orchestrating the execution of the project, managing training sessions, testing phases,
    and result generation with ease and efficiency.
  
- `main.py`: The central execution file of the project

## Study

All the theoretical background, study, and experiments conducted are documented in the 
**docs** folder in both English (`report.pdf`) and Italian (`relazione.pdf`).

## Collaborate

If you encounter any errors or have suggestions for improvements, we welcome collaboration and feedback from the 
community. You can contribute by:

- **Reporting Issues:** If you come across any bugs or issues, please submit them 
through GitHub's issue tracker for this project repository.

- **Pull Requests:** Feel free to submit pull requests with fixes, enhancements, 
or new features. We appreciate any contributions that improve the project.

Collaboration is essential for the continued development and improvement of this project. 
Let's work together to make it even better!

## References

- P. Alvarez Lopez, M. Behrisch, L. Bieker-Walz, J. Erdmann, Yun-Pang Flötteröd, R. Hilbrich, L. Lücken, J. Rummel, P. Wagner, E. Wiessner, (2018).  
  Microscopic Traffic Simulation using SUMO.  
  IEEE Intelligent Transportation Systems Conference (ITSC).  
  https://elib.dlr.de/124092/  

- Lucas N. Alegre, (2019).  
  SUMO-RL.  
  GitHub repository.  
  https://github.com/LucasAlegre/sumo-rl  

- A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann, (2021).  
  Stable-Baselines3: Reliable Reinforcement Learning Implementations.  
  Journal of Machine Learning Research.  
  http://jmlr.org/papers/v22/20-1364.html  

- R. Chimisso (2023).  
  2WSI-RL.  
  GitHub Repository.  
  https://github.com/rChimisso/2WSI-RL  

