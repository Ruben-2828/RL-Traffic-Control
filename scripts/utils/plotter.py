import os
from os import listdir
from os.path import join, isdir

import matplotlib.pyplot as plt
import pandas as pd
from re import split


class Plotter:

    # dict to set title according to set metric
    _Titles = {
        'system_total_stopped': 'Number of stationary vehicles',
        'system_total_waiting_time': 'Total waiting time',
        'system_mean_waiting_time': 'Mean waiting time',
        'system_mean_speed': 'Mean speed'
    }
    # dict to set y labels according to set metric
    _Ylabels = {
        'system_total_stopped': 'Stationary vehicles',
        'system_total_waiting_time': 'Waiting time',
        'system_mean_waiting_time': 'Waiting time',
        'system_mean_speed': 'Speed'
    }
    # dict containing the font settings for labels and title
    _labels_font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 18,
    }

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
        if input_path.endswith('.csv'):
            self.csv_files.append(input_path)
        elif isdir(input_path):
            self.csv_files.extend(join(input_path, f) for f in listdir(input_path)
                                  if join(input_path, f).endswith('.csv'))

    def read_csvs(self) -> None:
        """
        read_csvs loads the content of the csv in the dataframe collection
        """
        for csv in self.csv_files:
            self.df.append(pd.read_csv(csv))

    def build_plot(self, out_folder: str = None) -> None:
        """
        build_plot builds a plot using data from the dataframe collection.
        Configs must be set before calling this.
        :param out_folder: folder to save the plots into. If not None, It will be appended
                           to self.output
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

            plt.title(self._Titles[metric], fontdict=self._labels_font)
            plt.xlabel('step', fontdict=self._labels_font)
            plt.ylabel(self._Ylabels[metric], fontdict=self._labels_font)

            for data in self.df:
                plt.plot(data.get('step'), data.get(metric))

            # Using regex split to remove file path and keep only file name
            plt.legend([split(r'[/\\]', item)[-1] for item in self.csv_files], loc="upper right")

            self._save_plot(fig, metric, out_folder)
            plt.close()

    def _save_plot(self, fig: plt.figure, metric: str, out_folder: str) -> None:
        """
        save_plot saves the plot following the format in input
        :param fig: figure to save
        :param metric: metric used in plot. Needed to set file name
        :param out_folder: folder in which to save the plots
        """
        if out_folder is not None:
            out_path = os.path.join(self.output, out_folder)
        else:
            out_path = self.output

        os.makedirs(out_path, exist_ok=True)

        output_file = out_path + '/' + metric
        fig.savefig(output_file, dpi=96)

    def clear(self) -> None:
        """
        clear_csv_files clears the csv files in input
        """
        self.csv_files = []
        self.df = []
