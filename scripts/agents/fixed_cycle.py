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

    def run(self, learn: bool, out_path: str) -> str:
        """
        Run agents for number of episodes specified in self.config['Runs'] and save the csvs
        :param learn: if True, agent will learn. Value DOESN'T matter with Fixed Cycle
        :param out_path: path to save the csv file
        :return: path containing the csv output files
        """

        out_path = os.path.join(out_path, self.name)
        out_file = os.path.join(out_path, self.name)

        for curr_run in range(self.config['Runs']):
            done = False
            self.env.reset()
            while not done:
                done = self._step()
            self.env.save_csv(out_file, curr_run)

        self.env.close()

        return out_path

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

    def _step(self) -> bool:
        """
        Perform one step of the custom
        :return: bool, whether the simulation has terminated
        """
        for _ in range(self.env.delta_time):
            self.env._sumo_step()
        self.env._compute_observations()
        self.env._compute_rewards()
        self.env._compute_info()
        return self.env._compute_dones()['__all__']
