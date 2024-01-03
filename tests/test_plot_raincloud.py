import unittest

import seaborn as sns

from issych.figure import prepare_ax, plot_raincloud


class TestPlotRaincloud(unittest.TestCase):
    def test(self):
        data = sns.load_dataset('tips')
        fig, ax = prepare_ax(path_ini='tests/testdata/config/rcparams.ini')
        plot_raincloud(data, ax, x='day', y='total_bill', box=True)
        fig.savefig('test_plot_raincloud.png')