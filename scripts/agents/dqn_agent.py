import os
from typing import Union

import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from sumo_rl import SumoEnvironment

from scripts.agents.learning_agent import LearningAgent
from scripts.utils.config_values import Metric


class DQNAgent(LearningAgent):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        """
        DQN Agent constructor
        :param config: dict containing the configuration of the DQN agent
        :param env: Sumo Environment object
        :param name: name of the agent, used for saving models, csvs and plots
        """
        super().__init__(config, env, name)

    def _init_agent(self):
        """
        Initialize the agent object using self.config
        """

        self.agent = DQN(
            env=self.env,
            policy="MlpPolicy",
            learning_rate=self.config["Alpha"],
            learning_starts=0,
            train_freq=(1, 'step'),
            target_update_interval=100,
            gradient_steps=-1,
            gamma=self.config['Gamma'],
            exploration_fraction=self.config['Exp_fraction'],
            exploration_initial_eps=self.config['Init_epsilon'],
            exploration_final_eps=self.config['Final_epsilon'],
            verbose=0,
            device='auto'
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
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        for curr_run in range(self.config['Runs']):

            if learn:
                rows = []

                # total_timesteps are the env total steps, which are total time / time per step
                self.agent.learn(total_timesteps=self.env.sim_max_time // self.env.delta_time,
                                 callback=SaveInfos(rows))

                df = pd.DataFrame.from_records(rows, columns={'step'}.union(Metric))
                df.to_csv(out_file + "_ep" + str(curr_run) + ".csv")
            else:
                done = False
                state = self.env.reset()[0]
                while not done:
                    state, _, _, done, _ = self.env.step(self.agent.predict(state)[0])

                self.env.save_csv(out_file, curr_run)
        self.env.close()

        return out_path

    def save(self, path: str) -> None:
        """
        Saves the trained agent to a file
        :param path: path to save the trained agent to
        """
        self.agent.save(path)

    def load(self, path: str, env: SumoEnvironment) -> None:
        """
        Loads an agent from a file
        :param path: path to load the trained agent from
        :param env: new environment to run the loaded agent on
        """
        self.env = env
        self.agent = DQN.load(path, env=env)


class SaveInfos(BaseCallback):
    """
    Custom callback to save env infos after each step
    """
    def __init__(self, rows: list, verbose=0):
        """
        Class constructor
        :param rows: list to save infos to
        :param verbose: verbosity level,
                        0 -> no output, 1 -> info messages, 2 -> debug messages
        """
        super().__init__(verbose)
        self.rows = rows

    def _on_step(self) -> bool:
        """
        Method executed after each step to save infos
        :return: True to continue simulation, False to stop simulation
        """
        locals()
        self.rows.append(self.locals['infos'][0])
        return True
