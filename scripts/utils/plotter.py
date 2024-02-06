from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import pandas as pd

Titles = {
    'system_total_stopped': 'Number of stationary vehicles',
    'system_total_waiting_time': 'Total waiting time',
    'system_mean_waiting_time': 'Mean waiting time',
    'system_mean_speed': 'Mean speed'
}

Ylabels = {
    'system_total_stopped': 'Stationary vehicles',
    'system_total_waiting_time': 'Waiting time',
    'system_mean_waiting_time': 'Waiting time',
    'system_mean_speed': 'Speed'
}

labels_font = {
    'family': 'serif',
    'color':  'darkred',
    'weight': 'normal',
    'size': 18,
}


class Plotter:

    def __init__(self, output, metric, width=3840, height=1080):
        """
        Plotter builder
        :param output: path to output file
        :param metric: metric to use for plotting data
        :param width: output image width in pixels
        :param height: output image height in pixels
        """
        self.output = output
        self.height = height / 96
        self.width = width / 96
        self.df = []
        self.csv_files = []
        self.metric = metric
        self.fig = plt.figure(figsize=(self.width, self.height))

    def add_csv(self, input_path) -> None:
        """
        add_csv adds one (or more) csv file to the csv collection
        :param input_path: can be a dir with multiple csv files or a single csv file
        """
        if isfile(input_path):
            self.csv_files.append(input_path)
        else:
            self.csv_files.extend(join(input_path, f) for f in listdir(input_path)
                                  if join(input_path, f).endswith('.csv'))

    def read_csvs(self) -> None:
        """
        read_csvs loads the content of the csv in the dataframe collection
        """
        for csv in self.csv_files:
            self.df.append(pd.read_csv(csv))

    def build_plot(self) -> None:
        """
        build_plot builds a plot using data from the dataframe collection
        """
        self.read_csvs()

        plt.rcParams['lines.linewidth'] = 1
        plt.xlabel("Step", fontdict=labels_font)
        plt.ylabel(Ylabels[self.metric], fontdict=labels_font)
        plt.title(Titles[self.metric], fontdict=labels_font)

        for data in self.df:
            plt.plot(data.get('step'), data.get(self.metric))

        plt.legend(self.csv_files, loc="upper right")

    def clear_plot(self) -> None:
        """
        clear_plot clears the figure stored in the plotter
        """
        self.fig.clear()

    def save_plot(self) -> None:
        """
        save_plot saves the plot following the format in input
        """
        output_file = self.output + '_' + self.metric
        self.fig.savefig(output_file, dpi=96)

    def set_metric(self, metric) -> None:
        """
        set_metric sets the metric of the plotter
        :param metric: metric to use for plotting data
        """
        self.metric = metric
