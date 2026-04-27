import unittest

import numpy as np
import pandas as pd
from rpy2.robjects import r

from issych.stat import (
    convert_to_iqrs, nanzscore2value, iqr2value, value2nanzscore,
    arcsine_sqrt, kwargs4r)


class TestIQR(unittest.TestCase):
    def check_with_nan(self, SEQ, IQRS_Z):
        self.assertEqual(
            (pd.Series(convert_to_iqrs(SEQ)) - pd.Series(IQRS_Z)).sum(), 0)
        self.assertEqual(nanzscore2value(0, SEQ), 5.)
        self.assertEqual(value2nanzscore(5, SEQ), 0.)

    def test(self):
        IQRS = [-0.5, -0.25, 0., 0., 0., 0., 0., 0.25, 0.5]
        IQRS_Z = [-0.5, -0.25, 0., 0., 0., float('nan'), 0., 0., 0.25, 0.5]

        SEQ = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(list(convert_to_iqrs(SEQ)), IQRS)
        for iqr, v in zip(IQRS, SEQ):
            if iqr == 0:
                self.assertEqual(iqr2value(iqr, SEQ), 5.)
                continue
            self.assertEqual(iqr2value(iqr, SEQ), v)
        self.assertEqual(iqr2value(-0.5, SEQ), 1.)
        self.assertEqual(nanzscore2value(0, SEQ), 5.)
        self.assertEqual(value2nanzscore(5, SEQ), 0.)

        # Case including np.nan
        SEQ = [1, 2, 3, 4, 5, float('nan'), 6, 7, 8, 9]
        self.check_with_nan(SEQ, IQRS_Z)

        # Case including pd.NA
        SEQ = pd.Series(SEQ).astype('Float64')
        self.check_with_nan(SEQ, IQRS_Z)


class TestArcSineSqrt(unittest.TestCase):
    def test(self):
        # Calculated with R
        self.assertEqual(arcsine_sqrt(0), 0)
        self.assertAlmostEqual(arcsine_sqrt(1), 1.570796, places=6)
        self.assertAlmostEqual(arcsine_sqrt(0.5), 0.7853982, places=6)
        with self.assertRaises(ValueError):
            arcsine_sqrt(-0.1)
        with self.assertRaises(ValueError):
            arcsine_sqrt(1.1)





class TestKwargs4R(unittest.TestCase):
    def test(self):
        r('''
          val1 <- 1
          myfunc <- function(val1, val2, comment) {
            paste0(comment, ": ", val1 + val2) }
          ''')

        kwargs = {'val1': '@val1', 'val2': 2, 'comment': 'test'}
        self.assertEqual(r(f'myfunc({kwargs4r(kwargs)})')[0], 'test: 3')
        kwargs = {'key': [1, 2, 'a']}
        self.assertEqual(kwargs4r(kwargs), 'key = c(1, 2, "a")')
        kwargs = {'key': {'val1': 1, 'val2': 2}}
        self.assertEqual(kwargs4r(kwargs), 'key = c("val1" = 1, "val2" = 2)')
