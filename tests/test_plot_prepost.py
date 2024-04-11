import unittest

import pingouin as pg

from issych.figure import prepare_ax, plot_prepost


class TestPlotPrepost(unittest.TestCase):
    def test(self):
        data = pg.read_dataset('rm_anova_wide')
        fig, ax = prepare_ax(path_ini='tests/testdata/config/rcparams.ini')
        plot_prepost(data, ax, pre='Before', post='1 week')

        fig.savefig('test_plot_prepost.png')