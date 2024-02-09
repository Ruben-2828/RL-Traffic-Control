
import yaml

from utils.config_values import Metric, TrafficType, AgentType


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
        parse read the configs from the yaml file in input and sets plotter and
        learning agents configs
        """

        with open(self.yaml_file) as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        if 'Plotter_settings' not in configs or 'Agent_settings' not in configs:
            raise Exception('bad config file format')

        temp_plotter_config = configs['Plotter_settings']
        temp_la_configs = configs['Agent_settings']

        if not (self.check_plotter_config(temp_plotter_config) and
                self.check_learning_agents_config(temp_la_configs)):
            raise Exception('bad config file format')

        self.plotter_config = temp_plotter_config
        self.learning_agents_instances = temp_la_configs

    def check_plotter_config(self, configs: dict) -> bool:
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

    def check_learning_agents_config(self, configs: dict) -> bool:
        """
        checks if the learning agents configs read from the yaml file are well formatted
        :param configs: dict representing the learning agents configs
        :return: True if learning agents configs are valid, False otherwise
        """

        if 'Traffic_type' not in configs:
            return False
        if configs['Traffic_type'] not in TrafficType:
            return False

        if 'Output' not in configs:
            return False

        if 'Instances' not in configs:
            return False
        for instance in configs['Instances'].values():
            if 'Agent_type' not in instance:
                return False
            if instance['Agent_type'] not in AgentType:
                return False
            if 'Runs' not in instance:
                return False
            if instance['Runs'] <= 0:
                return False
            if 'Alpha' not in instance:
                return False
            if not (0 < instance['Alpha'] <= 1):
                return False
            if 'Gamma' not in instance:
                return False
            if not (0 <= instance['Gamma'] <= 1):
                return False
            if 'Init_epsilon' not in instance:
                return False
            if not (0 <= instance['Init_epsilon'] <= 1):
                return False
            if 'Min_epsilon' not in instance:
                return False
            if not (0 <= instance['Min_epsilon'] <= 1):
                return False
            if 'Decay' not in instance:
                return False
            if not (0 <= instance['Decay'] <= 1):
                return False

        return True



