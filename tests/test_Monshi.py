import unittest

import pandas as pd

from issych.fileio import load_config

from issych.monshi import Monshi


class TestMonshi(unittest.TestCase):
    def test(self):
        config = load_config('tests/testdata/config/monshi.ini')
        answer_sheet = pd.read_csv('tests/testdata/monshi/testdata.csv')
        answer_sheet = answer_sheet.applymap(
            lambda x: x if pd.isna(x) else x.split('.')[0])
        monshi = Monshi(answer_sheet).label(config.range).score(config)
        scores = monshi.get_scores()
        monshi = Monshi(answer_sheet).label(config.range).score(config)  # Check not destructing vars
        scores = monshi.get_scores()
        manual_scores = pd.read_csv(
            'tests/testdata/monshi/testdata-manually-scored.csv',
            index_col=['sub', 'timestamp'])
        self.assertTrue(scores.equals(manual_scores.astype('Float64')))