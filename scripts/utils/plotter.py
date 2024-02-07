import os
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

    def __init__(self, output=None, metrics=None, width=3840, height=1080):
        """
        Plotter builder
        :param output: path to output file
        :param metrics: metric to use for plotting data
        :param width: output image width in pixels
        :param height: output image height in pixels
        """
        self.output = output
        self.height = height / 96
        self.width = width / 96
        self.df = []
        self.csv_files = []
        self.metrics = metrics

    def set_configs(self, configs) -> None:
        """
        Sets plotter configurations
        :param configs: dict representing plotter configurations
        """
        self.output = configs['Output']
        if 'Height' in configs:
            self.height = configs['Height'] / 96
        if 'Width' in configs:
            self.width = configs['Width'] / 96
        self.metrics = configs['Metrics']

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
        build_plot builds a plot using data from the dataframe collection.
        Configs must be set before calling this.
        """
        if len(self.csv_files) == 0:
            raise ValueError('No csv files set')
        if self.metrics is None:
            raise ValueError('No metrics set')
        if self.output is None:
            raise ValueError('No output path set')

        self.read_csvs()

        # For each metric, build and save corresponding plot
        for metric in self.metrics:
            fig = plt.figure(figsize=(self.width, self.height))

            plt.rcParams['lines.linewidth'] = 1
            plt.title(Titles[metric], fontdict=labels_font)
            plt.xlabel('step')
            plt.ylabel(Ylabels[metric])
            plt.legend(self.csv_files, loc="upper right")

            for data in self.df:
                plt.plot(data.get('step'), data.get(metric))

            self.save_plot(fig, metric)
            plt.clf()

    def save_plot(self, fig, metric) -> None:
        """
        save_plot saves the plot following the format in input
        """
        # If output dir does not exist, create it
        os.makedirs(self.output, exist_ok=True)

        output_file = self.output + '/plot_' + metric
        fig.savefig(output_file, dpi=96)
