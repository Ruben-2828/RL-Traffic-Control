import pickle

from sumo_rl import SumoEnvironment
from scripts.custom.custom_true_online_sarsa import TrueOnlineSarsaLambdaDecay
from scripts.agents.sarsa_agent import SarsaAgent


class SarsaDecayAgent(SarsaAgent):

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

    def save(self, path: str) -> None:
        """
        Saves the trained agent to a file
        :param path: path to save the trained agent to
        """
        super().save(path)

        with open(path, 'rb') as f:
            data = pickle.load(f)
            data['decay'] = self.agent.decay

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str, env: SumoEnvironment) -> None:
        """
        Loads an agent from a file
        :param path: path to load the trained agent from
        :param env: new custom to run the loaded agent on
        """
        super().load(path, env)

        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.agent.decay = data['decay']

