import os

from sumo_rl import SumoEnvironment
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from scripts.agents.learning_agent import LearningAgent


class SarsaAgent(LearningAgent):
    
    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        """
        SARSA-Learning Agent constructor
        :param config: dict containing the configuration of the SARSA agent
        :param env: Sumo Environment object
        :param name: name of the agent, used for saving models, csvs and plots
        """
        super().__init__(config, env, name)

    def _init_agent(self):
        """
        Initialize the agent object using self.config
        """
        self.agent = TrueOnlineSarsaLambda(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            alpha=self.config['Alpha'],
            gamma=self.config['Gamma'],
            epsilon=self.config['Epsilon'],
            fourier_order=self.config['FourierOrder'],
            lamb=self.config['Lambda']
        )

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
            obs, _ = self.env.reset()
            terminated, truncated = False, False
            
            while not (terminated or truncated):
                action = self.agent.act(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action=action)

                self.agent.learn(state=obs, action=action, reward=reward, next_state=next_obs, done=terminated)
                obs = next_obs

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
