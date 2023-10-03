import unittest

import pandas as pd

from issych.fileio import load_config
from issych.dataframe import fullform_index, set_sdgt

from issych.dataframe import flatten_multi_index


class TestFMI(unittest.TestCase):
    def test_basic(self):
        table = pd.read_csv(
            'tests/testdata/dataframe/resulttable.csv', index_col=[0, 1, 2],
            dtype={'n': 'Int64'})
        flatten_multi_index(table)

    def test_appearance_setting_suite(self):
        table = pd.read_csv(
            'tests/testdata/dataframe/resulttable.csv', index_col=[0, 1, 2],
            dtype={'n': 'Int64'})

        sdgts = load_config('tests/testdata/config/dataframe.ini').sdgt
        table = set_sdgt(table, sdgts,
                         cols_rate=['rate'], cols_p=['p', 'padj'], thr_p=0.001)

        abbr = load_config('tests/testdata/config/dataframe.ini').abbr
        table = fullform_index(table, abbr)

        flatten_multi_index(table, pad=' ')
