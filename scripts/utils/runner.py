from sumo_rl import SumoEnvironment

import os

from scripts.agents.ql_agent import QLearningAgent


class Runner:
    """
    Runner allows to run multiple tests on agents with different configurations
    """

    def __init__(self, configs, plotter, learn=True):
        """
        Runner builder
        :param configs: dict representing runner configurations
        :param plotter: plotter object to plot results
        :param learn: boolean, if True agents will learn, if False agents will
                      only be tested
        """
        self.configs = configs
        self.plotter = plotter
        self.learn = learn
        self.agents = []
        self.env = self.create_environment(self.configs['Traffic_type'])

    def create_environment(self, traffic_type) -> SumoEnvironment:

        route_file = None

        if traffic_type == 'low':
            route_file = "big-intersection/BI_50_test.rou.xml"
        if traffic_type == 'medium':
            route_file = "big-intersection/BI_100_test.rou.xml"
        if traffic_type == 'high':
            route_file = "big-intersection/BI_200_test.rou.xml"

        return SumoEnvironment(
            net_file="big-intersection/BI.net.xml",
            route_file=route_file,
            use_gui=False,
            num_seconds=1000,
            min_green=10,
            max_green=50,
            single_agent=True,
            add_per_agent_info=False
        )

    def run(self) -> None:

        if not self.agents:
            self.load_agents_from_configs()

        traffic_type = self.configs['Traffic_type']
        output_path = os.path.join(self.configs['Output'], traffic_type)

        for agent in self.agents:
            print("Running agent: " + agent.get_name())
            agent.run(self.env, self.learn, output_path)

    def load_agents_from_configs(self):

        for name, config in self.configs['Instances'].items():
            agent = QLearningAgent(config, self.env, name)
            self.agents.append(agent)

    def save_agent_to_file(self):
        return None

    def load_agent_from_file(self):
        return None
