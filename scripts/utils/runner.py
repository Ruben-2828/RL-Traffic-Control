
import os

from scripts.agents.dqn_agent import DQNAgent
from scripts.agents.fixed_cycle import FixedCycleAgent
from scripts.agents.learning_agent import LearningAgent
from scripts.agents.ql_agent import QLearningAgent
from scripts.agents.sarsa_agent import SarsaAgent
from scripts.agents.sarsa_decay_agent import SarsaDecayAgent

from scripts.custom.custom_environment import CustomEnvironment
from scripts.utils.plotter import Plotter




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
        self._set_environment()

    def _set_environment(self) -> None:
        """
        custom setter, sets up custom from configs
        """
        env_config = self.configs['Environment']
        route_file = None

        if env_config['Traffic_type'] == 'low':
            route_file = "big-intersection/BI_50_test.rou.xml"
        if env_config['Traffic_type'] == 'medium':
            route_file = "big-intersection/BI_100_test.rou.xml"
        if env_config['Traffic_type'] == 'high':
            route_file = "big-intersection/BI_200_test.rou.xml"

        self.env = CustomEnvironment(
            route_file=route_file,
            gui=env_config['Gui'],
            num_seconds=env_config['Num_seconds'],
            min_green=env_config['Min_green'],
            max_green=env_config['Max_green'],
            yellow_time=env_config['Yellow_time'],
            delta_time=env_config['Delta_time'],
        )

    def run(self) -> None:
        """
        method to run all the agents simulations
        """
        if self.env is None:
            self._set_environment()
        if not self.agents:
            self._load_agents()

        traffic_type = self.configs['Environment']['Traffic_type']
        output_path = os.path.join(self.configs['Output_csv'], traffic_type)
        output_csvs_paths: dict[str, str] = {}

        for agent in self.agents:
            print("\nRunning agent: " + agent.get_name())
            csvs_path = agent.run(self.learn, output_path)
            output_csvs_paths[agent.get_name()] = csvs_path

        self._plot_per_agent(output_csvs_paths)
        self._plot_last_episode(output_csvs_paths)
        self._save_agents_to_file()

    def _plot_per_agent(self, csvs_paths: dict[str, str]) -> None:
        """
        for each agent, plot the results of its episodes using self.plotter
        :param csvs_paths: dict containing the agent and its path to csv files
        """
        for name, path in csvs_paths.items():
            self.plotter.add_csv(path)
            self.plotter.build_plot(name)
            self.plotter.clear()

    def _plot_last_episode(self, csvs_path: dict[str, str]) -> None:
        """
        plots results of last episode of each agent using self.plotter
        :param csvs_path: dict containing the agent and its path to csv files
        """
        for path in csvs_path.values():
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            last_episode = os.path.join(path, csv_files[-1])
            self.plotter.add_csv(last_episode)

        self.plotter.build_plot('last_episodes')
        self.plotter.clear()

    def _load_agents(self):
        """
        load (untrained) agents from config file and appends them to the agents list.
        if 'Model' field is in configs, it loads the agents from save files
        """
        for name, config in self.configs['Instances'].items():
            if config['Agent_type'] == 'QL':
                if 'Model' in config:
                    agent = QLearningAgent(config, None, name)
                    agent.load(config['Model'], self.env.get_sumo_env(False))
                else:
                    agent = QLearningAgent(config, self.env.get_sumo_env(False), name)
            if config['Agent_type'] == 'DQN':
                if 'Model' in config:
                    agent = DQNAgent(config, None, name)
                    agent.load(config['Model'], self.env.get_sumo_env(False))
                else:
                    agent = DQNAgent(config, self.env.get_sumo_env(False), name)
            if config['Agent_type'] == 'SARSA':
                if 'Model' in config:
                    agent = SarsaAgent(config, self.env.get_sumo_env(False), name)
                    agent.load(config['Model'], self.env.get_sumo_env(False))
                else:
                    agent = SarsaAgent(config, self.env.get_sumo_env(False), name)
            if config['Agent_type'] == 'SARSA_decay':
                if 'Model' in config:
                    agent = SarsaDecayAgent(config, self.env.get_sumo_env(False), name)
                    agent.load(config['Model'], self.env.get_sumo_env(False))
                else:
                    agent = SarsaDecayAgent(config, self.env.get_sumo_env(False), name)
            if config['Agent_type'] == 'FIXED':
                agent = FixedCycleAgent(config, self.env.get_sumo_env(True), name)
            self.agents.append(agent)

    def _save_agents_to_file(self) -> None:

        os.makedirs(self.configs['Output_model'], exist_ok=True)

        for agent in self.agents:
            out_file = self.configs['Output_model'] + '/' + agent.get_name() + '.pkl'
            agent.save(out_file)
