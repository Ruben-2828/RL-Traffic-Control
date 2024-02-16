from scripts.utils.runner import Runner
from scripts.utils.config_parser import ConfigsParser

from scripts.utils.plotter import Plotter

LIBSUMO_AS_TRACI = 1

config_parser = ConfigsParser('configs/test.yaml')
config_parser.parse()
plotter_configs = config_parser.get_plotter_config()
runner_configs = config_parser.get_runner_config()

p = Plotter()
p.set_configs(plotter_configs)

r = Runner(runner_configs, p)
r.run()

