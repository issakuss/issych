import unittest
from pathlib import Path

from dynaconf.vendor.tomllib import TOMLDecodeError
from dynaconf import Dynaconf
import pandas as pd

from issych.dataclass import Dictm, Pathm


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

    def test_drop(self):
        dictm = Dictm({'a': 1, 'b': 2, 'c': 3, 'd': 4})
        dictm1 = dictm.drop('a')
        self.assertEqual(list(dictm1.keys()), ['b', 'c', 'd'])
        dictm2 = dictm1.drop(['b', 'c'])
        self.assertEqual(list(dictm2.keys()), ['d'])

        self.assertEqual(dictm.drop('e', skipnk=True).keys(), dictm.keys())
        self.assertEqual(list(dictm.drop(['d', 'e'], skipnk=True).keys()),
                         ['a', 'b', 'c'])
        with self.assertRaises(RuntimeError):
            dictm.drop('e')

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

class TestPathm(unittest.TestCase):
    def test_misc(self):
        pd.read_csv(Pathm('tests/testdata/pathm/pathm1/file.csv'))
        pd.read_csv(Pathm('tests/testdata/pathm/{dr}/file.csv')(dr='pathm1'))

    def test_truediv(self):
        self.assertTrue((Pathm() / 'issych').exists())
        self.assertTrue((Pathm() / Path('issych')).exists())
        self.assertTrue((Pathm() / Pathm('issych')).exists())

    def test_path_method(self):
        self.assertTrue(Pathm().resolve().exists())
        self.assertTrue((Pathm() / 'issych').resolve().exists())

    def test_template(self):
        mypath = Pathm('tests/{foo}/pathm/{bar}/file')
        self.assertTrue(mypath(foo='testdata', bar='pathm1').exists())
        self.assertTrue(mypath(foo='testdata', bar='pathm2').exists())
        self.assertFalse(mypath(foo='testdata', bar='pathm3').exists())
