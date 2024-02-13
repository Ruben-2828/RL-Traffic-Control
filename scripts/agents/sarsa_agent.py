import os
import numpy as np

from sumo_rl import SumoEnvironment
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from scripts.agents.learning_agent import LearningAgent

class SarsaAgent(LearningAgent):
    
    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        """
        Q-Learning Agent constructor
        :param config: dict containing the configuration of the QL agent
        :param env: Sumo Environment object
        """
        super().__init__(config, env, name)

    def init_agent(self):
        
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
        if self.agent is None:
            self.init_agent()

        for curr_run in range(self.config['Runs']):
            obs, info = env.reset()
            terminated, truncated = False, False
            
            while not (terminated or truncated):
                action = self.agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action=action)
                
                if isinstance(obs, tuple):  
                    obs_array = np.array(obs) 
                else:
                    obs_array = obs
                    
                if isinstance(next_obs, tuple):  
                    next_obs_array = np.array(next_obs)  
                else:
                    next_obs_array = next_obs
                
                if learn:
                    self.agent.learn(state=obs_array, action=action, reward=reward, next_state=next_obs_array, done=terminated)

                obs = next_obs

            out_file = os.path.join(out_path, self.name, self.name)
            env.save_csv(out_file, curr_run)
            env.reset()


    def save(self) -> None:
        pass

    def load(self) -> None:
        pass