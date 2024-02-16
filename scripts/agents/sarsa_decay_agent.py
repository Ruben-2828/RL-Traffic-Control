import os
import pickle

from sumo_rl import SumoEnvironment
from scripts.custom.custom_true_online_sarsa import TrueOnlineSarsaLambdaDecay
from scripts.agents.learning_agent import LearningAgent


class SarsaDecayAgent(LearningAgent):

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
        self.agent = TrueOnlineSarsaLambdaDecay(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            alpha=self.config['Alpha'],
            gamma=self.config['Gamma'],
            epsilon=self.config['Epsilon'],
            decay=self.config['Decay'],
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

    def save(self, path: str) -> None:
        """
        Saves the trained agent to a file
        :param path: path to save the trained agent to
        """
        data = {
            'alpha': self.agent.alpha,
            'gamma': self.agent.gamma,
            'epsilon': self.agent.epsilon,
            'decay': self.agent.decay,
            'lamb': self.agent.lamb,
            'fourier_order': self.agent.basis.order,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str, env: SumoEnvironment) -> None:
        """
        Loads an agent from a file
        :param path: path to load the trained agent from
        :param env: new custom to run the loaded agent on
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.env = env

        self.agent = TrueOnlineSarsaLambdaDecay(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            alpha=data['alpha'],
            gamma=data['gamma'],
            epsilon=data['epsilon'],
            decay=data['decay'],
            fourier_order=data['fourier_order'],
            lamb=data['lamb']
        )
