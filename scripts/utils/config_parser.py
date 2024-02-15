
import yaml

from scripts.utils.config_values import Metric, TrafficType, AgentType


class ConfigsParser:
    """
    ConfigsParser is used to read configuration files (in yaml format) and
    to check the config format to avoid errors during learning phase.
    It only checks if the mandatory parameters are set, if there are other
    useless parameters, they won't be considered at all.
    """

    def __init__(self, yaml_file: str):
        """
        ConfigsParser class builder
        :param yaml_file: path to yaml config file
        """
        self.yaml_file = yaml_file
        self.plotter_config = None
        self.learning_agents_instances = None

    def get_plotter_config(self) -> dict:
        """
        Returns the plotter configs
        :return: dict representing the plotter configs
        """
        return self.plotter_config

    def get_runner_config(self) -> dict:
        """
        Returns the learning agents configs
        :return: dict representing the learning agents configs
        """
        return self.learning_agents_instances

    def parse(self) -> None:
        """
        read the configs from the yaml file in input and sets plotter and learning agents configs
        """

        with open(self.yaml_file) as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        if 'Plotter_settings' not in configs or 'Agent_settings' not in configs:
            raise Exception('bad config file format')

        temp_plotter_config = configs['Plotter_settings']
        temp_la_configs = configs['Agent_settings']

        if not (self._check_plotter_config(temp_plotter_config) and
                self._check_learning_agents_config(temp_la_configs)):
            raise Exception('bad config file format')

        self.plotter_config = temp_plotter_config
        self.learning_agents_instances = temp_la_configs

    def _check_plotter_config(self, configs: dict) -> bool:
        """
        checks if the plotter configs read from the yaml file are well formatted
        :param configs: dict representing the plotter configs
        :return: True if plotter configs are valid, False otherwise
        """

        if 'Output' not in configs:
            return False
        if 'Metrics' not in configs:
            return False
        if not all(item in Metric for item in configs['Metrics']):
            return False

        return True

    def _check_learning_agents_config(self, configs: dict) -> bool:
        """
        checks if the learning agents configs read from the yaml file are well formatted
        :param configs: dict representing the learning agents configs
        :return: True if learning agents configs are valid, False otherwise
        """
        if 'Output_csv' not in configs:
            return False
        if 'Output_model' not in configs:
            return False

        if 'Environment' not in configs:
            return False
        if not self._check_environment(configs['Environment']):
            return False

        if 'Instances' not in configs:
            return False
        for instance in configs['Instances'].values():
            if 'Agent_type' not in instance:
                return False
            if instance['Agent_type'] not in AgentType:
                return False
            if instance['Agent_type'] == 'QL':
                return self._check_ql(instance)
            if instance['Agent_type'] == 'DQN':
                return self._check_dqn(instance)
            if instance['Agent_type'] == 'SARSA':
                return self._check_sarsa(instance)
            if instance['Agent_type'] == 'FIXED':
                return self._check_fixed(instance)
        return False

    def _check_environment(self, config: dict) -> bool:
        """
        Checks if config represents a valid environment
        :param config: dict representing the config
        :return: True if valid, False otherwise
        """

        if 'Traffic_type' not in config:
            return False
        if config['Traffic_type'] not in TrafficType:
            return False
        if 'Gui' not in config:
            return False
        if 'Num_seconds' not in config:
            return False
        if config['Num_seconds'] <= 0:
            return False
        if 'Min_green' not in config:
            return False
        if config['Min_green'] <= 0:
            return False
        if 'Max_green' not in config:
            return False
        if config['Max_green'] < config['Min_green']:
            return False
        if 'Yellow_time' not in config:
            return False
        if config['Yellow_time'] <= 0:
            return False
        if 'Delta_time' not in config:
            return False
        if config['Delta_time'] < config['Yellow_time']:
            return False

        return True

    def _check_ql(self, config: dict) -> bool:
        """
        Checks if config represents a valid QL agent
        :param config: dict representing the config
        :return: True if valid, False otherwise
        """

        if 'Runs' not in config:
            return False
        if config['Runs'] <= 0:
            return False
        if 'Model' not in config:
            if 'Alpha' not in config:
                return False
            if not (0 < config['Alpha'] <= 1):
                return False
            if 'Gamma' not in config:
                return False
            if not (0 <= config['Gamma'] <= 1):
                return False
            if 'Init_epsilon' not in config:
                return False
            if not (0 <= config['Init_epsilon'] <= 1):
                return False
            if 'Min_epsilon' not in config:
                return False
            if not (0 <= config['Min_epsilon'] <= 1):
                return False
            if 'Decay' not in config:
                return False
            if not (0 <= config['Decay'] <= 1):
                return False

        return True

    def _check_dqn(self, config: dict) -> bool:
        """
        Checks if config represents a valid DQN agent
        :param config: dict representing the config
        :return: True if valid, False otherwise
        """

        if 'Runs' not in config:
            return False
        if config['Runs'] <= 0:
            return False
        if 'Alpha' not in config:
            return False
        if not (0 < config['Alpha'] <= 1):
            return False
        if 'Gamma' not in config:
            return False
        if not (0 <= config['Gamma'] <= 1):
            return False
        if 'Init_epsilon' not in config:
            return False
        if not (0 <= config['Init_epsilon'] <= 1):
            return False
        if 'Final_epsilon' not in config:
            return False
        if not (0 <= config['Final_epsilon'] <= 1):
            return False
        if 'Exp_fraction' not in config:
            return False
        if not (0 <= config['Exp_fraction'] <= 1):
            return False

        return True

    def _check_sarsa(self, config: dict) -> bool:
        """
        Checks if config represents a valid SARSA agent
        :param config: dict representing the config
        :return: True if valid, False otherwise
        """

        if 'Runs' not in config:
            return False
        if config['Runs'] <= 0:
            return False
        if 'Alpha' not in config:
            return False
        if not (0 < config['Alpha'] <= 1):
            return False
        if 'Gamma' not in config:
            return False
        if not (0 <= config['Gamma'] <= 1):
            return False
        if 'Epsilon' not in config:
            return False
        if not (0 <= config['Epsilon'] <= 1):
            return False
        if 'FourierOrder' not in config:
            return False
        if 'Lambda' not in config:
            return False

        return True

    def _check_fixed(self, config: dict) -> bool:
        """
        Checks if config represents a valid Fixed Cycle agent
        :param config: dict representing the config
        :return: True if valid, False otherwise
        """

        if 'Runs' not in config:
            return False
        if config['Runs'] <= 0:
            return False

        return True



