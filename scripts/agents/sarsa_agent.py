import os
import numpy as np

from sumo_rl import SumoEnvironment
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from scripts.agents.learning_agent import LearningAgent


def to_numpy_array(arr) -> np.ndarray:
    """
    Converts the input to a numpy array if it isn't already
    :param arr: array to be converted
    :return: array in numpy format
    """
    return np.array(arr) if isinstance(arr, tuple) else arr


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

    def run(self, env: SumoEnvironment, learn: bool, out_path: str) -> None:
        """
        Run agents for number of episodes specified in self.config['Runs'] and save the csvs
        :param env: Sumo Environment object
        :param learn: if True, agent will learn
        :param out_path: path to save the csv file
        """
        if self.agent is None:
            self._init_agent()

        for curr_run in range(self.config['Runs']):
            obs, _ = env.reset()
            terminated, truncated = False, False
            
            while not (terminated or truncated):
                action = self.agent.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action=action)
                
                obs_array = to_numpy_array(obs)
                next_obs_array = to_numpy_array(next_obs)
                
                if learn:
                    self.agent.learn(state=obs_array,
                                     action=action,
                                     reward=reward,
                                     next_state=next_obs_array,
                                     done=terminated)

                obs = next_obs

            out_file = os.path.join(out_path, f"{self.name}/{self.name}")
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
