import unittest

from issych.stat import convert_to_iqrs, nanzscore, nanzscore2value


class TestIQR(unittest.TestCase):
    def test(self):
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        convert_to_iqrs(seq)
        nanzscore(seq)
        nanzscore2value(0, seq)