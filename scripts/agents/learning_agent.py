
from abc import ABC, abstractmethod
from sumo_rl import SumoEnvironment


class LearningAgent(ABC):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        """
        LearningAgent constructor
        :param config: dict containing the configuration of the agent
        """
        self.config = config
        self.env = env
        self.agent = None
        self.name = name

    def get_name(self) -> str:
        return self.name

    @abstractmethod
    def run(self, env: SumoEnvironment, learn: bool, out_path: str) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass
