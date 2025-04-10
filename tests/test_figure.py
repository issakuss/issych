import unittest

import pandas as pd
import seaborn as sns

from issych.dataclass import Dictm
from issych.figure import (
    prepare_ax, prepare_axes, mask_area, plot_within, plot_raincloud,
    SigMarker, plot_corrmat)


IN_PATH_TOML = 'tests/testdata/config/rcparams.toml'

class TestPlotWithin(unittest.TestCase):
    def test(self):
        data = (sns.load_dataset('exercise')
                .pivot(index='id', columns='time', values='pulse'))
        fig, ax = prepare_ax(in_path_toml=IN_PATH_TOML)
        plot_within(data, ax=ax, x='1 min', y='30 min')
        fig.savefig('tests/output/test_plot_within.png')


class TestMaskArea(unittest.TestCase):
    def test(self):
        data = sns.load_dataset('fmri')
        fig, ax = prepare_ax(in_path_toml=IN_PATH_TOML)
        sns.lineplot(x="timepoint", y="signal", data=data, ax=ax)
        mask_area(3.5, 7.5, orient='h', color='gray', ax=ax, zorder=-1)
        mask_area(12.5, 12.5, orient='h', ax=ax, ec='white')
        mask_area(0.08, 0.12, orient='vert', ax=ax, color='red')
        fig.savefig('tests/output/test_mask_area.png')


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


class TestSigMarker(unittest.TestCase):
    def test(self):
        data = sns.load_dataset('tips')
        fig, axes = prepare_axes(n_col=2, axsizeratio=(.5, 1.),
                                 in_path_toml=IN_PATH_TOML)

        sns.barplot(data=data, x='day', y='total_bill', hue='sex', ax=axes[0])
        marker = SigMarker(axes[0])
        marker.mark('patches', 0, 4, comment='1')
        marker.mark('xticks', 0, 1, comment='2')
        marker.mark('xticks', 0, 3, comment='3')
        axes[0].legend_.remove()

        custom_thrs = {1.1: 'n.s.', 0.01: '<0.01'}

        sns.barplot(data=data, x='day', y='total_bill', hue='sex', ax=axes[1])
        marker = SigMarker(axes[1], coef_interval_btw_layer=0.3)
        marker.sigmark('xticks', 0, 2, p_value=0.50)
        marker.sigmark('xticks', 1, 3, p_value=0.05)
        marker.sigmark('patches', 3, 5, p_value=0.009)
        marker.sigmark('patches', 6, 7, p_value=0.2, thresholds=custom_thrs)
        marker.sigmark('patches', 5, 7, p_value=0.001, thresholds=custom_thrs)
        axes[1].legend_.remove()

        fig.tight_layout()
        fig.savefig('tests/output/test_sigmarker.png')


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
