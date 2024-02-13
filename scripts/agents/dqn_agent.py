import os
from typing import Union

import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from sumo_rl import SumoEnvironment

from scripts.agents.learning_agent import LearningAgent
from scripts.utils.config_values import Metric
from stable_baselines3.common.env_checker import check_env

class DQNAgent(LearningAgent):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        """
        DQN Agent constructor
        :param config: dict containing the configuration of the DQN agent
        :param env: Sumo Environment object
        """
        super().__init__(config, env, name)

    def init_agent(self):

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

    def run(self, env: SumoEnvironment, learn: bool, out_path: str) -> None:

        if self.agent is None:
            self.init_agent()

        out_file = os.path.join(out_path, self.name, self.name)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        for curr_run in range(self.config['Runs']):

            if learn:
                rows = []

                # Time steps are the env total steps, which are total time / time per step
                self.agent.learn(total_timesteps=self.env.sim_max_time // self.env.delta_time,
                                 callback=TestCallback(rows))

                df = pd.DataFrame.from_records(rows, columns={'step'}.union(Metric))
                df.to_csv(out_file + "_ep" + str(curr_run) + ".csv")
            else:
                done = False
                state = env.reset()[0]
                while not done:
                    state, _, _, done, _ = env.step(self.agent.predict(state)[0])

                env.save_csv(out_file, curr_run)

            env.reset()

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass


class TestCallback(BaseCallback):
    def __init__(self, rows: list, verbose=0):
        super().__init__(verbose)
        self.rows = rows

    def _on_step(self):
        locals()
        self.rows.append(self.locals['infos'][0])
        return True
