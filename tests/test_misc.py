import unittest
from time import sleep
import tempfile
from pathlib import Path

from dynaconf.vendor.tomllib import TOMLDecodeError
from dynaconf import Dynaconf

from tqdm import tqdm
from joblib import Parallel, delayed

from issych.misc import (
    Pval, Float, Dictm, alphabet2num, meas_exectime, tqdm_joblib)



class TestAlphabet2Num(unittest.TestCase):
    def test(self):
        self.assertEqual(alphabet2num('A'), 1)
        self.assertEqual(alphabet2num('a'), 1)
        self.assertEqual(alphabet2num('Z'), 26)
        self.assertEqual(alphabet2num('AA'), 27)


class TestMeasExecTime(unittest.TestCase):
    def test(self):
        """
        Test the meas_exectime decorator.
        """
        @meas_exectime
        def foo_():
            sleep(1)
            return 1
        foo_()

class TqdmJobLib(unittest.TestCase):
    def test(self):
        def process_item(item):
            sleep(0.1)
            return item * 2
        items = list(range(10))
        paralleled = (delayed(process_item)(item) for item in items)
        with tqdm_joblib(tqdm(total=len(items))) as pbar:
            results = Parallel(n_jobs=-1)(paralleled)
        self.assertEqual(sorted(results), [item * 2 for item in items])


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
                       'b': {'ba': 3, 'bb': 4},
                       'c': 5})
        flatten = dictm.flatten()
        self.assertEqual(flatten.aa, 1)
        self.assertEqual(flatten.bb, 4)
        self.assertEqual(flatten.c, 5)

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
            settings_files='tests/testdata/config/misc.toml'))
        self.assertEqual(dictm.level1.key1, 'value1')
        self.assertEqual(dictm.level1.key2, ['value2', 'value3'])
        self.assertEqual(dictm.level1.key3, 3)
        self.assertEqual(dictm.level1.level2.key11, 'valuelevel2')
        self.assertEqual(dictm.level1.level2.level3.key111, 'valuelevel3')

    def test_dynacof_path(self):
        dictm = Dictm('tests/testdata/config/misc.toml')
        self.assertEqual(dictm.level1.key1, 'value1')
        self.assertEqual(dictm.level1.key2, ['value2', 'value3'])
        self.assertEqual(dictm.level1.key3, 3)
        self.assertEqual(dictm.level1.level2.key11, 'valuelevel2')
        self.assertEqual(dictm.level1.level2.level3.key111, 'valuelevel3')

        with self.assertRaises(FileNotFoundError):
            Dictm('path_not_exists.toml')
        with self.assertRaises(TOMLDecodeError):
            Dictm('tests/testdata/config/corrupted_toml.toml')

    def test_to_yaml(self):
        dictm = Dictm(pval=Pval(0.024), tval=Float(1.23))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output.yaml"
            dictm.to_yaml(out_path)
            
            with open(out_path, "r", encoding="utf-8") as f:
                yaml_str = f.read()
                
            self.assertIn("pval: !Pval 0.024", yaml_str)
            self.assertIn("tval: !Float 1.23", yaml_str)
        
        # Test nested dictm
        nested_dictm = Dictm(level1=Dictm(pval=Pval(0.05)))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path_nested = Path(tmpdir) / "output_nested.yaml"
            nested_dictm.to_yaml(out_path_nested)
            
            with open(out_path_nested, "r", encoding="utf-8") as f:
                yaml_str_nested = f.read()
                
            self.assertIn("level1:", yaml_str_nested)
            self.assertIn("pval: !Pval 0.05", yaml_str_nested)
