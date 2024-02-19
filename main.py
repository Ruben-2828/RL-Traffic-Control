from scripts.utils.runner import Runner
from scripts.utils.config_parser import ConfigsParser
from scripts.utils.plotter import Plotter


# In this example of code, we use the config file learn_low.yaml to train agents on low traffic

config_parser = ConfigsParser('configs/learn_low.yaml')
config_parser.parse()

p = Plotter()
p.set_configs(config_parser.get_plotter_config())

r = Runner(config_parser.get_runner_config(), p)
r.run()
