import unittest

from issych.misc import alphabet_to_num


class TestAlphabetToNum(unittest.TestCase):
    def test(self):
        self.assertEqual(alphabet_to_num('A'), 1)
        self.assertEqual(alphabet_to_num('a'), 1)
        self.assertEqual(alphabet_to_num('Z'), 26)
        self.assertEqual(alphabet_to_num('AA'), 27)