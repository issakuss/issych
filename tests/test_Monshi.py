import unittest
from pathlib import Path

import pandas as pd

from issych.dataclass import Dictm
from issych.dataframe import cast_to_nullable
from issych.monshi import Monshi, CantReverseError, score_as_monfig


class TestMonshi(unittest.TestCase):
    def setUp(self):
        in_dir_test = Path('tests/testdata')
        self.in_path_data = in_dir_test / 'monshi/testdata.csv'
        answer_sheet = pd.read_csv(self.in_path_data)
        self.answer_sheet = (
            answer_sheet.astype(str)
            .map(lambda x: x if pd.isna(x) else x.split('.')[0]))

        self.in_path_monfig = in_dir_test / 'config/monfig.toml'
        self.monfig = Dictm(self.in_path_monfig)
        self.monshi = Monshi(self.answer_sheet)

        in_path_manual_replaced = in_dir_test / \
                                  'monshi/testdata-manually-replaced.csv'
        self.manually_replaced = (pd.read_csv(in_path_manual_replaced)
                                  .set_index(['sub', 'timestamp'])
                                  .astype('Float64')
                                  .reset_index())
        in_path_manual_score = in_dir_test / \
                               'monshi/testdata-manually-scored.csv'
        self.manually_scored = (pd.read_csv(in_path_manual_score)
                                .set_index(['sub', 'timestamp'])
                                .astype('Float64')
                                .reset_index())

    def test_separate(self):
        with self.assertRaises(RuntimeError):
            self.monshi.separate({'dummy': ['1', '2', 3]})
        self.monshi.separate(self.monfig._cols_item)
        tester = pd.testing.assert_frame_equal
        tester(self.monshi.info, self.answer_sheet.iloc[:, [0, 1]])
        tester(self.monshi.scale1, self.answer_sheet.iloc[:, [2, 3, 4, 6, 7]])
        tester(self.monshi.scale2, self.answer_sheet.iloc[:, 8:18])
        tester(self.monshi.scale3, self.answer_sheet.iloc[:, 18:21])

    def test_score(self):
        self.assertTrue(self.answer_sheet.equals(self.monshi.get_sheet()))
        with self.assertRaises(RuntimeError):  # Run .score() before .separate() run
            self.monshi.score(self.monfig)
        self.monshi.separate(self.monfig._cols_item)

        invalid_monfig = Dictm('tests/testdata/config/invalid_monfig.toml')
        with self.assertRaises(CantReverseError):  # idx_reverse without min_plus_max
            self.monshi.separate({'dummy_scale1': [9, 10]})
            self.monshi.score({'dummy_scale1': invalid_monfig['dummy_scale1']})
        with self.assertRaises(CantReverseError):  # Negative subscale value without min_plus_max
            self.monshi.separate({'dummy_scale2': [9, 10]})
            self.monshi.score({'dummy_scale2': invalid_monfig['dummy_scale2']})
        with self.assertRaises(RuntimeError):  # Without necessary questionnaire setting
            self.monshi.separate({'dummy_scale3': [9, 10]})
            self.monshi.score({'dummy_scale1': invalid_monfig['dummy_scale1']})

        self.monshi.separate(self.monfig._cols_item)
        self.assertTrue(self.answer_sheet.drop(['ignored'], axis=1)
                        .equals(self.monshi.get_sheet()))
        self.monshi.replace_choices(self.monfig)
        current_sheet = (self.monshi.get_sheet()
                         .set_index(['sub', 'timestamp'])
                         .astype('float64').reset_index())
        with self.assertRaises(ValueError):
            self.monshi.validate(
                self.monfig | {'scale2': {'validation': {'min_answer': 2}}})
        with self.assertRaises(ValueError):  # validation failed
            self.monshi.validate(
                self.monfig | {'scale2': {'validation': {'max_answer': 4}}})
        self.assertTrue(current_sheet.equals(
            self.manually_replaced.set_index(['sub', 'timestamp'])
            .astype('float64').reset_index()))
        scores = self.monshi.score(self.monfig)
        self.assertTrue(scores.equals(self.manually_scored))

    def test_score_as_monfig(self):
        scores = score_as_monfig(self.answer_sheet, self.monfig)
        self.assertTrue(scores.equals(self.manually_scored))

        scores = score_as_monfig(self.answer_sheet, self.in_path_monfig)
        self.assertTrue(scores.equals(self.manually_scored))
