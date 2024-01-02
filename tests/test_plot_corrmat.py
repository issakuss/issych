import unittest

import pandas as pd
import seaborn as sns

from issych.figure import load_config, plot_corrmat


class TestPlotCorrmat(unittest.TestCase):
    def test(self):
        data = sns.load_dataset('iris')
        data = pd.get_dummies(data, prefix='iris', dtype=int)
        abbr = load_config('tests/testdata/config/abbr.ini').flatten()
        g = plot_corrmat(data, path_ini='tests/testdata/config/rcparams.ini',
                         abbr=abbr, kwargs_diag={'lw': 0.5})
        g.savefig('test_plot_corrmat.png')