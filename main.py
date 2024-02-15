
from scripts.utils.runner import Runner
from scripts.utils.config_parser import ConfigsParser

from scripts.utils.plotter import Plotter

config_parser = ConfigsParser('configs/config_sarsa.yaml')
config_parser.parse()
plotter_configs = config_parser.get_plotter_config()
runner_configs = config_parser.get_runner_config()

p = Plotter()
p.set_configs(plotter_configs)

r = Runner(runner_configs, p)
r.run()
