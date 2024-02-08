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
        seconds=1000,
        route="big-intersection/BI_50_test.rou.xml",
        runs=10
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
    subprocess.run(['python', 'scripts/ql_runner.py', '-a', str(args.alpha), '-g', str(args.gamma), '-d', str(args.decay), '-fixed', str(args.fixed), '-s', str(args.seconds), '-route', str(args.route), '-runs', str(args.runs)])

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
    study_name = 'ql_hyperparameters_optimization'  
    study = optuna.create_study(study_name=study_name, directions=['maximize', 'minimize', 'minimize', 'minimize'])
    study.optimize(objective, n_trials=10) #n_trials = number of hyperparameters combinations

    all_trials = study.trials

    # Write YAML file
    yaml_output_path = "configs/config_ql_from_hp-opt.yaml"
    with open(yaml_output_path, 'w') as yaml_output_file:
        yaml_output_data = {
            'Plotter_settings': {
                'Output': 'output/plots/ql',
                'Width': 3840,
                'Height': 1080,
                'Metrics': ['system_total_stopped', 'system_total_waiting_time', 'system_mean_waiting_time', 'system_mean_speed']
            },
            'Agent_settings': {
                'Traffic_type_training': ['low', 'medium', 'high'],
                'Instances': {}
            }
        }
        for idx, trial in enumerate(all_trials, start=1):
            trial_data = {
                'Alpha': round(trial.params['alpha'], 5),
                'Gamma': round(trial.params['gamma'], 5),
                'Decay': round(trial.params['decay'], 5),
                'Metrics': dict(zip(['system_mean_speed', 'system_mean_waiting_time', 'system_total_stopped', 'system_total_waiting_time'], trial.values))
            }
            yaml_output_data['Agent_settings']['Instances'][f'QL_run_{idx}'] = trial_data
        yaml.dump(yaml_output_data, yaml_output_file, default_flow_style=False)

    print('Results saved to', yaml_output_path)
