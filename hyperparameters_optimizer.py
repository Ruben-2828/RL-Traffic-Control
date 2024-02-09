import optuna
import os
import argparse
import subprocess
import yaml
import glob
import numpy as np

def objective(trial):
    # Hyperparameters to be optimized
    alpha_search = trial.suggest_float('alpha', 0.0001, 0.1)
    gamma_search = trial.suggest_float('gamma', 0.75, 0.999)
    decay_search = trial.suggest_float('decay', 0.9, 0.9999)

    # Run simulation 
    args = argparse.Namespace(
        alpha=alpha_search,
        gamma=gamma_search,
        decay=decay_search,
        fixed=False,
        seconds=10000,
        route="big-intersection/BI_50_test.rou.xml",
        runs=5
    )
    results = run_simulation(args)

    system_mean_speeds = []
    system_mean_waiting_times = []
    system_total_stoppeds = []
    system_total_waiting_times = []

    # Extract the metrics from the YAML files
    for result in results:
        system_mean_speeds.append(result['system_mean_speed'])
        system_mean_waiting_times.append(result['system_mean_waiting_time'])
        system_total_stoppeds.append(result['system_total_stopped'])
        system_total_waiting_times.append(result['system_total_waiting_time'])

    # Return the aggregated metrics to be optimized
    return np.mean(system_mean_speeds), -np.mean(system_mean_waiting_times), -np.mean(system_total_stoppeds), -np.mean(system_total_waiting_times)

def run_simulation(args):
    subprocess.run(['python', 'main.py', '-a', str(args.alpha), '-g', str(args.gamma), '-d', str(args.decay), '-fixed', str(args.fixed), '-s', str(args.seconds), '-route', str(args.route), '-runs', str(args.runs)])

    # Extract from YAML files
    yaml_output_folder = "output/yaml"
    yaml_files = glob.glob(os.path.join(yaml_output_folder, "values_BI_QL_conn0_ep*.yaml"))

    all_last_iteration_values = []
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as file:
            last_iteration_values = yaml.safe_load(file)
            all_last_iteration_values.append(last_iteration_values)

    return all_last_iteration_values

if __name__ == '__main__':
    study = optuna.create_study(directions=['maximize', 'minimize', 'minimize', 'minimize'])
    study.optimize(objective, n_trials=10)

    best_trials = study.best_trials

    print('All best trials:')
    # Print the best trials truncated to 5 decimal places for readability reasons
    for idx, best_trial in enumerate(best_trials, start=1):
        print('Hyperparameters:', {key: round(value, 5) for key, value in best_trial.params.items()})
        print('Metrics:', {key: round(value, 5) for key, value in dict(zip(['mean_system_mean_speed', 'mean_negative_system_mean_waiting_time', 'mean_negative_system_total_stopped', 'mean_negative_system_total_waiting_time'], best_trial.values)).items()})

