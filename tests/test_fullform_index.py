import unittest

import pandas as pd

from issych.fileio import load_config

from issych.dataframe import fullform_index


class TestFullformIndex(unittest.TestCase):
    def test(self):
        table = pd.read_csv(
            'tests/testdata/dataframe/resulttable.csv', index_col=[0, 1, 2],
            dtype={'n': 'Int64'})
        abbr = load_config('tests/testdata/config/dataframe.ini').abbr
        fullform_index(table, abbr, for_index=False, for_columns=False)
        fullform_index(table, abbr)
