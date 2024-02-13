
from scripts.utils.runner import Runner
from scripts.utils.config_parser import ConfigsParser

from scripts.utils.plotter import Plotter

config_parser = ConfigsParser('configs/test.yaml')
config_parser.parse()
plotter_configs = config_parser.get_plotter_config()
runner_configs = config_parser.get_runner_config()


#r = Runner(runner_configs, None)
#r.run()

p = Plotter()
p.set_configs(plotter_configs)
#p.add_csv('output/csv/test/high/QL_run_1/QL_run_1_conn0_ep4.csv')
p.add_csv('output/csv/test/low/QL_run_2/QL_run_2_conn0_ep4.csv')
#p.add_csv('output/csv/test/high/DQN_run_1/DQN_run_1_ep4.csv')
p.add_csv('output/csv/test/low/DQN_run_2/DQN_run_2_ep4.csv')
p.build_plot()
