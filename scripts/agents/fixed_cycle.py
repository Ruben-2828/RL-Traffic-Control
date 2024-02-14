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

    def run(self, env: SumoEnvironment, learn: bool, out_path: str) -> None:
        """
        Run agents for number of episodes specified in self.config['Runs'] and save the csvs
        :param env: Sumo Environment object
        :param learn: if True, agent will learn. Value DOESN'T matter with Fixed Cycle
        :param out_path: path to save the csv file
        """
        env.reset()

        for curr_run in range(self.config['Runs']):

            # DA RIFARE DIOOOOOOOOOOOOOOOOOOOOOO

            done = False
            while env.sim_step < 10000:
                env._sumo_step()

            out_file = os.path.join(out_path, self.name, self.name)
            env.save_csv(out_file, curr_run)
            env.reset()

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
