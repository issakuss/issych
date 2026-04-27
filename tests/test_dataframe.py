import unittest

import numpy as np
import pandas as pd

from issych.dataframe import vec2sqmatrix
from numpy.testing import assert_array_equal


class TestVec2Sqmatrix(unittest.TestCase):
    def test(self):
        seq1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        seq2 = [-1.4, 0.78, -112.4, 111.5]
        mat2 = [[-1.4, 0.78], [-112.4, 111.5]]
        seq3 = [1, 2, 3, 4, 5]
        assert_array_equal(vec2sqmatrix(seq1), np.array(mat1))
        assert_array_equal(vec2sqmatrix(seq2), np.array(mat2))
        with self.assertRaises(RuntimeError):
            vec2sqmatrix(seq3)

class TestCastToNullable(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, np.nan],
            'bool_col': [True, False, True],
            'obj_col': ['a', 'b', None],
            'na_int_col': pd.Series([1, 2, pd.NA], dtype="Int64"),
            'na_float_col': pd.Series([1.1, pd.NA, 3.3], dtype="Float64"),
            'na_bool_col': pd.Series([True, pd.NA, False], dtype="boolean"),
            'unchanged': pd.date_range("2024-01-01", periods=3),
        })

    def _test_cast(self, result: pd.DataFrame):
        self.assertEqual(str(result['int_col'].dtype), 'Int64')
        self.assertEqual(str(result['float_col'].dtype), 'Float64')
        self.assertIs(result['float_col'][2], pd.NA)
        self.assertEqual(str(result['bool_col'].dtype), 'boolean')
        self.assertEqual(str(result['int_col'].dtype), 'Int64')
        self.assertEqual(str(result['float_col'].dtype), 'Float64')
        self.assertEqual(str(result['bool_col'].dtype), 'boolean')
        self.assertTrue(np.issubdtype(result['unchanged'].dtype, np.datetime64))
