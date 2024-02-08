import optuna
import os
import argparse
import subprocess
import yaml
import glob
import numpy as np

def objective(trial):
    # Define the hyperparameters to be optimized
    alpha = trial.suggest_float('alpha', 0.0001, 0.1)
    gamma = trial.suggest_float('gamma', 0.75, 0.999)
    decay = trial.suggest_float('decay', 0.9, 0.9999)

    # Run simulation with given hyperparameters
    args = argparse.Namespace(
        alpha=alpha,
        gamma=gamma,
        decay=decay,
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

    for result in results:
        system_mean_speeds.append(result['system_mean_speed'])
        system_mean_waiting_times.append(result['system_mean_waiting_time'])
        system_total_stoppeds.append(result['system_total_stopped'])
        system_total_waiting_times.append(result['system_total_waiting_time'])

    # Return the aggregated metrics to be optimized
    return np.mean(system_mean_speeds), -np.mean(system_mean_waiting_times), -np.mean(system_total_stoppeds), -np.mean(system_total_waiting_times)

def run_simulation(args):
    # Run your simulation script with given arguments
    subprocess.run(['python', 'main.py', '-a', str(args.alpha), '-g', str(args.gamma), '-d', str(args.decay), '-fixed', str(args.fixed), '-s', str(args.seconds), '-route', str(args.route), '-runs', str(args.runs)])

    # Extract the last iteration values from the YAML file
    yaml_output_folder = "output/yaml"
    yaml_files = glob.glob(os.path.join(yaml_output_folder, "values_BI_QL_conn0_ep*.yaml"))

    all_last_iteration_values = []
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as file:
            last_iteration_values = yaml.safe_load(file)
            all_last_iteration_values.append(last_iteration_values)

    return all_last_iteration_values

if __name__ == '__main__':
    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(directions=['maximize', 'minimize', 'minimize', 'minimize'])
    study.optimize(objective, n_trials=100)

    best_trials = study.best_trials

    print('All trials:')
    for trial in best_trials:
        print(f'Hyperparameters: {trial.params}')
        print(f'Metrics: {trial.values}')
