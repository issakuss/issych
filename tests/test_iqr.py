import unittest

import pandas as pd
from issych.stat import (
    convert_to_iqrs, nanzscore, nanzscore2value, value2nanzscore)


class TestIQR(unittest.TestCase):
    def check_with_nan(self, SEQ, IQRS_Z, Z):
        self.assertEqual(
            (pd.Series(convert_to_iqrs(SEQ)) - pd.Series(IQRS_Z)).sum(), 0)
        self.assertEqual(nanzscore2value(0, SEQ), 5.)
        self.assertAlmostEqual((nanzscore(SEQ) - pd.Series(Z)).sum(), 0)
        self.assertEqual(value2nanzscore(5, SEQ), 0.)

    def test(self):
        IQRS = [-0.5, -0.25, 0., 0., 0., 0., 0., 0.25, 0.5]
        IQRS_Z = [-0.5, -0.25, 0., 0., 0., float('nan'), 0., 0., 0.25, 0.5]
        NANZ = [-1.54919334, -1.161895, -0.77459667, -0.38729833, 0.,
                float('nan'),0.38729833, 0.77459667, 1.161895, 1.54919334]
        Z = [-1.54919334, -1.161895, -0.77459667, -0.38729833, 0.,
             0.38729833, 0.77459667, 1.161895, 1.54919334]

        SEQ = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(list(convert_to_iqrs(SEQ)), IQRS)
        self.assertEqual(nanzscore2value(0, SEQ), 5.)
        self.assertAlmostEqual((nanzscore(SEQ) - pd.Series(Z)).sum(), 0)
        self.assertEqual(value2nanzscore(5, SEQ), 0.)
        with self.assertRaises(ValueError):
            nanzscore(pd.DataFrame(SEQ))

        # Case including np.nan
        SEQ = [1, 2, 3, 4, 5, float('nan'), 6, 7, 8, 9]
        self.check_with_nan(SEQ, IQRS_Z, NANZ)

        # Case including pd.NA
        SEQ = pd.Series(SEQ).astype('Float64')
        self.check_with_nan(SEQ, IQRS_Z, NANZ)