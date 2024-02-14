from sumo_rl import SumoEnvironment

import os

from scripts.agents.dqn_agent import DQNAgent
from scripts.agents.fixed_cycle import FixedCycleAgent
from scripts.agents.learning_agent import LearningAgent
from scripts.agents.ql_agent import QLearningAgent
from scripts.utils.plotter import Plotter

from scripts.agents.sarsa_agent import SarsaAgent


class Runner:
    """
    Runner class that allows to run multiple tests on agents with different configurations
    """

    def __init__(self, configs: dict, plotter: Plotter, learn: bool = True):
        """
        Runner constructor
        :param configs: dict representing runner configurations
        :param plotter: plotter object to plot results
        :param learn: boolean, if True agents will learn, if False it won't
        """
        self.configs: dict = configs
        self.plotter: Plotter = plotter
        self.learn: bool = learn
        self.agents: [LearningAgent] = []
        self.env: SumoEnvironment = None

    def _set_environment(self, traffic_type: str) -> None:
        """
        environment setter
        :param traffic_type: type of traffic of the environment
        """
        route_file = None

        if traffic_type == 'low':
            route_file = "big-intersection/BI_50_test.rou.xml"
        if traffic_type == 'medium':
            route_file = "big-intersection/BI_100_test.rou.xml"
        if traffic_type == 'high':
            route_file = "big-intersection/BI_200_test.rou.xml"

        self.env = SumoEnvironment(
            net_file="big-intersection/BI.net.xml",
            route_file=route_file,
            use_gui=False,
            num_seconds=10000,
            min_green=5,
            max_green=50,
            single_agent=True,
            add_per_agent_info=False,
            sumo_warnings=False,
            fixed_ts=True
        )

    def run(self) -> None:
        """
        method to run all the agents simulations
        """

        if not self.agents:
            self.load_agents_from_configs()
        if self.env is None:
            self._set_environment(self.configs['Traffic_type'])

        traffic_type = self.configs['Traffic_type']
        output_path = os.path.join(self.configs['Output'], traffic_type)

        for agent in self.agents:
            print("\nRunning agent: " + agent.get_name())
            agent.run(self.env, self.learn, output_path)

        self.env.close()
        self.env = None

    def load_agents_from_configs(self):
        """
        load (untrained) agents from config file and appends them to the agents list
        """

        for name, config in self.configs['Instances'].items():
            if config['Agent_type'] == 'QL':
                agent = QLearningAgent(config, self.env, name)
            if config['Agent_type'] == 'DQN':
                agent = DQNAgent(config, self.env, name)
            if config['Agent_type'] == 'SARSA':
                agent = SarsaAgent(config, self.env, name)
            if config['Agent_type'] == 'FIXED':
                agent = FixedCycleAgent(config, self.env, name)
            self.agents.append(agent)

    def load_agents_from_file(self, input_file: str) -> None:
        """
        Load (already trained) agent from file and appends it to the agents list
        :param input_file: str representing the agent's save file
        """
        return None
