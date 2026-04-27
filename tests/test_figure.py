from pathlib import Path
import unittest

import pandas as pd
import seaborn as sns

from issych.misc import Dictm
from issych.figure import (
    prepare_ax, prepare_axes, plot_within, plot_raincloud,
    plot_corrmat, set_rcparams)


IN_PATH_TOML = 'tests/testdata/config/rcparams.toml'


class TestSetrcparams(unittest.TestCase):
    def test(self):
        IN_DIR = Path('tests/testdata/config/')
        set_rcparams(in_path_toml=IN_DIR / 'rcparams-wo-sect1.toml')
        set_rcparams(in_path_toml=IN_DIR / 'rcparams-wo-sect2.toml')


class TestPlotWithin(unittest.TestCase):
    def test(self):
        data = (sns.load_dataset('exercise')
                .pivot(index='id', columns='time', values='pulse'))
        fig, ax = prepare_ax(in_path_toml=IN_PATH_TOML)
        plot_within(data, ax=ax, x='1 min', y='30 min')
        fig.savefig('tests/output/test_plot_within.png')


class TestPlotRainCloud(unittest.TestCase):
    def test(self):
        data = sns.load_dataset('tips')
        fig, axes = prepare_axes(3, 2, in_path_toml=IN_PATH_TOML)
        plot_raincloud(data, ax=axes[0, 0], x='day', y='total_bill')
        plot_raincloud(data.total_bill, ax=axes[1, 0],
                       kwargs_strip={'color': 'red'},
                       kwargs_box={'color': 'blue'})
        plot_raincloud(data.total_bill, ax=axes[2, 0], box=False)
        plot_raincloud(data.total_bill, ax=axes[0, 1], strip=False)
        plot_raincloud(data.total_bill, ax=axes[1, 1], strip=False, box=False)
        fig.savefig('tests/output/test_raincloud.png')


class TestPlotCorrmat(unittest.TestCase):
    def test1(self):
        data = sns.load_dataset('iris')
        data = pd.get_dummies(data, prefix='iris', dtype=int)
        g = plot_corrmat(data,
                         in_path_toml='tests/testdata/config/rcparams.toml')
        g.savefig('tests/output/test_plot_corrmat1.png')

    def test2(self):
        data = sns.load_dataset('iris')
        data = pd.get_dummies(data, prefix='iris', dtype=int)
        abbr = Dictm('tests/testdata/config/abbr.toml').general
        g = plot_corrmat(data, method='spearman', sdgt=4, abbr=abbr,
                         color_positive='red', color_negative='blue',
                         in_path_toml='tests/testdata/config/rcparams.toml')
        g.savefig('tests/output/test_plot_corrmat2.png')


if __name__ == '__main__':
    unittest.main()
