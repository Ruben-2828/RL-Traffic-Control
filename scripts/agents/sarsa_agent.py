import os
import numpy as np

from sumo_rl import SumoEnvironment

from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from linear_rl.fourier import FourierBasis
from scripts.agents.learning_agent import LearningAgent

from gym import spaces  

class TrueOnlineSarsaLambda(LearningAgent):

    def __init__(self, state_space: spaces.Space, action_space: spaces.Space, config: dict, env: SumoEnvironment, name: str):
        """
        Sarsa-Learning Agent constructor
        :param state_space: observation space of the environment
        :param action_space: action space of the environment
        :param config: dict containing the configuration of the SARSA agent
        :param env: Sumo Environment object
        """
        super().__init__(config, env, name)
        self.state_space = state_space
        self.action_space = action_space
        self.min_max_norm = True
        self.basis = FourierBasis(state_space=self.state_space, action_space=self.action_space, order=10)


    def init_agent(self):
        self.agent = TrueOnlineSarsaLambda(
            self.env.observation_space,
            self.env.action_space,
            alpha=self.config['Alpha'],
            gamma=self.config['Gamma'],
            epsilon=self.config['Epsilon'],
            fourier_order=7,
            lamb=0.95
    )


    def run(self, learn: bool, out_path: str) -> None:
        if self.agent is None:
            self.init_agent()

        for curr_run in range(self.config['Runs']):
            obs, info = self.env.reset()
            terminated, truncated = False, False

            while not (terminated or truncated):
                action = self.agent.act(obs)

                next_obs, reward, terminated, truncated, info = self.env.step(action=action)

                if learn:
                    self.agent.learn(state=obs, action=action, reward=reward, next_state=next_obs, done=terminated)

                obs = next_obs

            out_file = os.path.join(out_path, self.name, f"{self.name}_{curr_run}")
            self.env.save_csv(out_file, curr_run)
            self.env.reset()

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass

    def get_q_value(self, features, action):
        return np.dot(self.theta[action], features)
        
    def get_features(self, state):
        if self.min_max_norm:
            state = (state - self.state_space.low) / (self.state_space.high - self.state_space.low)
        return self.basis.get_features(state)
    
    def reset_traces(self):
        self.q_old = None
        for a in range(self.action_dim):
            self.et[a].fill(0.0)
    
    def act(self, obs):
        features = self.get_features(obs)
        return self.get_action(features)

    def get_action(self, features):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = [self.get_q_value(features, a) for a in range(self.action_dim)]
            return q_values.index(max(q_values))
        