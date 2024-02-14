from sumo_rl import SumoEnvironment
from scripts.agents.learning_agent import LearningAgent


class FixedCycleAgent(LearningAgent):

    def __init__(self, config: dict, env: SumoEnvironment, name: str) -> None:
        """
        Fixed Cycle Agent constructor
        :param config: dict containing the configuration of the Fixed Cycle agent
        :param env: Sumo Environment object
        """
        super().__init__(config, env, name)

    def run(self, env: SumoEnvironment, learn: bool, out_path: str) -> None:
        """
        Run the agent
        :param env: Sumo Environment object
        :param learn: bool, whether the agent should learn or not
        :param out_path: str, path to save the model
        """
        self.env = env
        self.env.reset()
        done = False
        while not done:
            done = self._step()
        self.env.reset()

    def save(self) -> None:
        """
        Save the agent's model
        """
        pass

    def load(self) -> None:
        """
        Load the agent's model
        """
        pass

    def _step(self) -> bool:
        """
        Perform one step of the environment
        :return: bool, whether the simulation has terminated
        """
        for _ in range(self.env.delta_time):
            self.env._sumo_step()
        self.env._compute_observations()
        self.env._compute_rewards()
        self.env._compute_info()
        return self.env._compute_dones()['__all__']
