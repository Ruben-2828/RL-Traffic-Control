
from sumo_rl import SumoEnvironment


class CustomEnvironment:
    """
    Class to bobobobobooboboboobobob TODO: commentare meglio
    """

    def __init__(self,
                 route_file: str,
                 gui: bool,
                 num_seconds: int,
                 min_green: int,
                 max_green: int,
                 yellow_time: int,
                 delta_time: int) -> None:
        """
        CustomEnvironment constructor
        :param route_file: Path to the route file
        :param gui: if True, graphical user interface is shown
        :param num_seconds: number of seconds to run the simulation for
        :param min_green: minimum time for green phase
        :param max_green: maximum time for green phase
        :param yellow_time: time for yellow phase
        :param delta_time: time used per step
        """
        self.route_file = route_file
        self.gui = gui
        self.num_seconds = num_seconds
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.delta_time = delta_time

    def get_sumo_env(self, fixed: bool) -> SumoEnvironment:
        """
        Get the sumo environment
        :param fixed: True for fixed cycle, False for learning agent
        :return: Corresponding SumoEnvironment
        """
        return SumoEnvironment(
            net_file="big-intersection/BI.net.xml",
            route_file=self.route_file,
            use_gui=self.gui,
            num_seconds=self.num_seconds,
            delta_time=self.delta_time,
            yellow_time=self.yellow_time,
            min_green=self.min_green,
            max_green=self.max_green,
            fixed_ts=fixed,
            add_per_agent_info=False,
            sumo_warnings=False,
            single_agent=True,
        )
