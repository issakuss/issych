import unittest

from issych.dataclass import Dictm


class TestDictm(unittest.TestCase):
    def test_basic(self):
        dictm = Dictm({'a': 1, 'b': 2})
        self.assertEqual(dictm.a, 1)
        self.assertIsNone(dictm.get('c'))

    def test_merge(self):
        dictm = Dictm({'a': 1, 'b': 2})
        dictm_to_add = Dictm({'c': 3})
        dictm = dictm | dictm_to_add
        self.assertEqual(dictm.a, 1)
        self.assertEqual(dictm.c, 3)

    def test_flatten(self):
        dictm = Dictm({'a': {'aa': 1, 'ab': 2},
                       'b': {'ba': 3, 'bb': 4}})
        flatten = dictm.flatten()
        self.assertEqual(flatten.aa, 1)

    def test_full(self):
        dictm = Dictm({'abbr': 'Full'})
        self.assertEqual(dictm.full('abbr'), 'Full')
        self.assertEqual(dictm.full('abbr_not_exist'), 'abbr_not_exist')