import os

from sumo_rl import SumoEnvironment     # type: ignore
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

from scripts.agents.learning_agent import LearningAgent


class FixedCycleAgent(LearningAgent):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        """
        Q-Learning Agent constructor
        :param config: dict containing the configuration of the QL agent
        :param env: Sumo Environment object
        """
        super().__init__(config, env, name)

    def run(self, env: SumoEnvironment, learn: bool, out_path: str) -> None:

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
        pass

    def load(self) -> None:
        pass
