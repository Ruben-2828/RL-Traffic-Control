from sumo_rl import SumoEnvironment
import os

from sumo_rl import SumoEnvironment     # type: ignore

from scripts.agents.learning_agent import LearningAgent


class FixedCycleAgent(LearningAgent):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        """
        Fixed cycle Agent constructor
        :param config: dict containing the configuration of the Fixed Cycle agent
        :param env: Sumo Environment object
        :param name: name of the agent, used for saving models, csvs and plots
        """
        super().__init__(config, env, name)

    def _init_agent(self):
        self.agent = None

    def run(self, learn: bool, out_path: str) -> None:
        """
        Run agents for number of episodes specified in self.config['Runs'] and save the csvs
        :param learn: if True, agent will learn. Value DOESN'T matter with Fixed Cycle
        :param out_path: path to save the csv file
        """

        for curr_run in range(self.config['Runs']):
            done = False
            self.env.reset()
            while not done:
                done = self._step()
            out_file = os.path.join(out_path, self.name, self.name)
            self.env.save_csv(out_file, curr_run)

        self.env.close()

    def save(self) -> None:
        """
        Saves the trained agent to a file
        """
        pass

    def load(self) -> None:
        """
        Loads an agent from a file
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
