import unittest
from time import sleep

from tqdm import tqdm
from joblib import Parallel, delayed

from issych.misc import alphabet2num, meas_exectime, tqdm_joblib


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
