import unittest

import seaborn as sns

from issych.figure import prepare_axes, plot_raincloud


class TestPlotRaincloud(unittest.TestCase):
    def test(self):
        data = sns.load_dataset('tips')
        fig, axes = prepare_axes(
            3, 1, path_ini='tests/testdata/config/rcparams.ini')
        plot_raincloud(data, axes[0], x='day', y='total_bill', box=True)
        plot_raincloud(data.total_bill, axes[1], box=True)
        plot_raincloud(data[['total_bill']], axes[2], box=True)
        fig.savefig('test_plot_raincloud.png')