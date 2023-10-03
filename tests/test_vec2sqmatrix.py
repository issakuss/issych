import unittest

import numpy as np

from issych.misc import vec2sqmatrix


class TestVec2Sqmatrix(unittest.TestCase):
    def test(self):
        seq1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        seq2 = [-1.4, 0.78, -112.4, 111.5]
        mat2 = [[-1.4, 0.78], [-112.4, 111.5]]
        seq3 = [1, 2, 3, 4, 5]
        self.assertTrue(np.array_equal(vec2sqmatrix(seq1), np.array(mat1)))
        self.assertTrue(np.array_equal(vec2sqmatrix(seq2), np.array(mat2)))
        with self.assertRaises(ValueError):
            vec2sqmatrix(seq3)