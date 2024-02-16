
from abc import ABC, abstractmethod
from sumo_rl import SumoEnvironment


class LearningAgent(ABC):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        """
        LearningAgent constructor
        :param config: dict containing the configuration of the agent
        :param env: Sumo Environment object
        :param name: name of the agent, used for saving models, csvs and plots
        """
        self.config = config
        self.env = env
        self.agent = None
        self.name = name

    @abstractmethod
    def _init_agent(self):
        """
        Initialize the agent object using self.config
        """
        pass

    def get_name(self) -> str:
        """
        Name of the agent getter
        :return: string containing the name of the agent
        """
        return self.name

    @abstractmethod
    def run(self, learn: bool, out_path: str) -> str:
        """
        Run agents for number of episodes specified in self.config['Runs'] and save the csvs
        :param learn: if True, agent will learn
        :param out_path: path to save the csv file
        :return: path containing the csv output files
        """
        pass

    def save(self, path: str) -> None:
        """
        Saves the trained agent to a file
        :param path: path to save the trained agent to
        """
        pass

    def load(self, path: str, env: SumoEnvironment) -> None:
        """
        Loads an agent from a file
        :param path: path to load the trained agent from
        :param env: new custom to run the loaded agent on
        """
        pass
