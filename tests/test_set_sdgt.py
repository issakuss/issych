import unittest

import pandas as pd

from issych.fileio import load_config

from issych.dataframe import set_sdgt


class TestSetSDGT(unittest.TestCase):
    def test(self):
        table = pd.read_csv(
            'tests/testdata/dataframe/resulttable.csv', index_col=[0, 1, 2],
            dtype={'n': 'Int64'})
        sdgts = load_config('tests/testdata/config/dataframe.ini').sdgt
        set_sdgt(table, sdgts)
        set_sdgt(table, sdgts, cols_rate=['rate', 'p'])
        set_sdgt(table, sdgts,
                 cols_rate=['rate'], cols_p=['p', 'padj'], thr_p=0.001)
