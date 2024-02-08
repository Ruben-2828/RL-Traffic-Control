import os
import sys
from utils.config_parser import ConfigsParser

from utils.plotter import Plotter

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

'''
if __name__ == "__main__":
    out_csv = f"output/BI"
    runs = 5
    fixed = False   # To run with fixed timing traffic signals

    env = SumoEnvironment(
        net_file="big-intersection/BI.net.xml",
        route_file="big-intersection/BI_50_test.rou.xml",
        out_csv_name=out_csv,
        use_gui=False,
        num_seconds=100000,
        min_green=10,
        max_green=50,
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=0.1,
                gamma=0.99,
                exploration_strategy=EpsilonGreedy(
                    initial_epsilon=0.05, min_epsilon=0.005, decay=1.0
                ),
            )
            for ts in env.ts_ids
        }

        done = {"__all__": False}
        infos = []
        if fixed:
            while not done["__all__"]:
                _, _, done, _ = env.step({})
        else:
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, _ = env.step(action=actions)

                for agent_id in ql_agents.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
        env.save_csv(out_csv, run)
        env.close()
'''

config_parser = ConfigsParser('configs/config_ql.yaml')
config_parser.parse()
plotter_configs = config_parser.get_plotter_config()

p = Plotter()
p.set_configs(plotter_configs)
p.add_csv('output/csv')
p.build_plot()
