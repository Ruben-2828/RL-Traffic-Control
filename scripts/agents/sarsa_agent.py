import os
from sumo_rl import SumoEnvironment
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from scripts.agents.learning_agent import LearningAgent
from sumo_rl.util.gen_route import write_route_file
from gym import spaces  

class SarsaLearningAgent(LearningAgent):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        """
        Sarsa-Learning Agent constructor
        :param config: dict containing the configuration of the SARSA agent
        :param env: Sumo Environment object
        """
        super().__init__(config, env, name)

    def init_agent(self):
        state_space = self.env.observation_space  
        action_space = self.env.action_space

        self.agent = TrueOnlineSarsaLambda(
            state_space=state_space,
            action_space=action_space,
            alpha=self.config['Alpha'],
            gamma=self.config['Gamma'],
            epsilon=self.config['Epsilon'],
            fourier_order=self.config.get('FourierOrder', 7),
            lamb=self.config.get('Lambda', 0.95)
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
