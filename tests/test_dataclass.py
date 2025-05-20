import unittest

from dynaconf.vendor.tomllib import TOMLDecodeError
from dynaconf import Dynaconf

from issych.dataclass import Dictm


class TestDictm(unittest.TestCase):
    def test_basic(self):
        dictm = Dictm(a=1, b=2)
        self.assertEqual(dictm.a, 1)
        self.assertIsNone(dictm.get('c'))

        dictm = Dictm({'a': 1, 'b': 2})
        self.assertEqual(dictm.a, 1)
        self.assertIsNone(dictm.get('c'))

        dictm = Dictm({'a': {'aa': {'aaa': 1}}})
        self.assertEqual(dictm.a.aa.aaa, 1)

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

    def test_may(self):
        dictm = Dictm({'abbr': 'Full'})
        self.assertEqual(dictm.may('abbr'), 'Full')
        self.assertEqual(dictm.may('abbr_not_exist'), 'abbr_not_exist')

    def test_dynaconf(self):
        dictm = Dictm(Dynaconf(
            settings_files='tests/testdata/config/dataclass.toml'))
        self.assertEqual(dictm.level1.key1, 'value1')
        self.assertEqual(dictm.level1.key2, ['value2', 'value3'])
        self.assertEqual(dictm.level1.key3, 3)
        self.assertEqual(dictm.level1.level2.key11, 'valuelevel2')
        self.assertEqual(dictm.level1.level2.level3.key111, 'valuelevel3')

    def test_dynacof_path(self):
        dictm = Dictm('tests/testdata/config/dataclass.toml')
        self.assertEqual(dictm.level1.key1, 'value1')
        self.assertEqual(dictm.level1.key2, ['value2', 'value3'])
        self.assertEqual(dictm.level1.key3, 3)
        self.assertEqual(dictm.level1.level2.key11, 'valuelevel2')
        self.assertEqual(dictm.level1.level2.level3.key111, 'valuelevel3')

        with self.assertRaises(FileNotFoundError):
            Dictm('path_not_exists.toml')
        with self.assertRaises(TOMLDecodeError):
            Dictm('tests/testdata/config/corrupted_toml.toml')
