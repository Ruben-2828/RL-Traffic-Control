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
        """
        super().__init__(config, env, name)

    def init_agent(self):

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

    def run(self, env: SumoEnvironment, learn: bool, out_path: str) -> None:

        if self.agent is None:
            self.init_agent()

        for curr_run in range(self.config['Runs']):

            done = False
            while not done:
                state, reward, _, done, _ = env.step(self.agent.act())
                if learn:
                    self.agent.learn(env.encode(state, env.ts_ids[0]), reward)

            out_file = os.path.join(out_path, self.name, self.name)
            env.save_csv(out_file, curr_run)
            env.reset()

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass
