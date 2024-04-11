import unittest

import pandas as pd

from issych.stat import rounddown


class TestSetSDGT(unittest.TestCase):
    def test(self):
        value = 3.1415926535
        self.assertEqual(rounddown(value, 2), 3.14)
        self.assertEqual(rounddown(value, 1), 3.1)
        self.assertEqual(rounddown(value, 0), 3)