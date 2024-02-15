import os

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

from scripts.agents.learning_agent import LearningAgent


class QLearningAgent(LearningAgent):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        """
        Q-Learning Agent constructor
        :param config: dict containing the configuration of the QL agent
        :param env: Sumo Environment object
        :param name: name of the agent, used for saving models, csvs and plots
        """
        super().__init__(config, env, name)

    def _init_agent(self):
        """
        Initialize the agent object using self.config
        """

        self.agent = QLAgent(
            starting_state=self.env.encode(self.env.reset()[0], self.env.ts_ids[0]),
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            alpha=self.config['Alpha'],
            gamma=self.config['Gamma'],
            exploration_strategy=EpsilonGreedy(initial_epsilon=self.config['Init_epsilon'],
                                               min_epsilon=self.config['Min_epsilon'],
                                               decay=self.config['Decay']
                                               ))

    def run(self, learn: bool, out_path: str) -> str:
        """
        Run agents for number of episodes specified in self.config['Runs'] and save the csvs
        :param learn: if True, agent will learn
        :param out_path: path to save the csv file
        :return: path containing the csv output files
        """
        if self.agent is None:
            self._init_agent()

        out_path = os.path.join(out_path, self.name)
        out_file = os.path.join(out_path, self.name)

        for curr_run in range(self.config['Runs']):

            done = False
            while not done:
                state, reward, _, done, _ = self.env.step(self.agent.act())
                if learn:
                    self.agent.learn(self.env.encode(state, self.env.ts_ids[0]), reward)

            self.env.save_csv(out_file, curr_run)
            self.env.reset()
        self.env.close()

        return out_path

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
