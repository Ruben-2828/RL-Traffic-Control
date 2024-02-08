import os
import glob
import sys
import yaml

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

from utils.config_parser import ConfigsParser
from utils.plotter import Plotter

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Save last iteration
def extract_last_iteration_values(csv_files, output_folder):
    for csv_file in csv_files:
        try:
            input_file_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_file_name = f"values_{input_file_name}.yaml"

            last_iteration_values = {}

            with open(csv_file, 'r') as f:
                lines = f.readlines()
                last_line = lines[-1].strip().split(',')
                last_iteration_values = {
                    "step": float(last_line[0]),
                    "system_total_stopped": int(last_line[1]),
                    "system_total_waiting_time": float(last_line[2]),
                    "system_mean_waiting_time": float(last_line[3]),
                    "system_mean_speed": float(last_line[4]),
                    "t_stopped": int(last_line[5]),
                    "t_accumulated_waiting_time": float(last_line[6]),
                    "t_average_speed": float(last_line[7]),
                    "agents_total_stopped": int(last_line[8]),
                    "agents_total_accumulated_waiting_time": float(last_line[9])
                }

            output_file_path = os.path.join(output_folder, output_file_name)
            with open(output_file_path, "w") as yamlfile:  
                yaml.dump(last_iteration_values, yamlfile)
                      
            print(f"Last iteration values saved successfully in: {output_file_path}")

        except Exception as e:
            print(f"An error occurred while processing {csv_file}: {e}")

if __name__ == "__main__":
    out_csv = f"output/csv/BI_QL"
    yaml_output_folder = f"output/yaml"
    runs = 10
    fixed = False   # To run with fixed timing traffic signals

    env = SumoEnvironment(
        net_file="big-intersection/BI.net.xml",
        route_file="big-intersection/BI_50_test.rou.xml",
        out_csv_name=out_csv,
        use_gui=False,
        num_seconds=1000,
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
        
        os.makedirs(yaml_output_folder, exist_ok=True)

    extracted_csv_files = glob.glob(out_csv+ "*.csv")
    grouped_csv_files = {}
    for csv_file in extracted_csv_files:
        episode_name = os.path.splitext(os.path.basename(csv_file))[0].split('_')[0]
        if episode_name not in grouped_csv_files:
            grouped_csv_files[episode_name] = []
        grouped_csv_files[episode_name].append(csv_file)

    for episode_name, csv_files in grouped_csv_files.items():
        extract_last_iteration_values(csv_files, yaml_output_folder)


config_parser = ConfigsParser('configs/config_ql.yaml')
config_parser.parse()
plotter_configs = config_parser.get_plotter_config()

p = Plotter()
p.set_configs(plotter_configs)
p.add_csv('output/csv')
p.build_plot()
