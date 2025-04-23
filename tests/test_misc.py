import unittest
from time import sleep

import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed

from issych.misc import (
    alphabet2num, meas_exectime, tqdm_joblib, group_by_feature_balance)


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


class TestGroupByFeatureBalance(unittest.TestCase):
    def setUp(self):
        def generate_toydata(rate_male: float, rate_med_exp: float,
                             mean_age: float, std_age: float, n_samples: int
                             ) -> pd.DataFrame:
            is_male = np.random.choice([True, False], size=n_samples,
                                       p=[rate_male, 1 - rate_male])
            has_med_exp = np.random.choice([True, False], size=n_samples,
                                           p=[rate_med_exp, 1 - rate_med_exp])
            age = np.random.normal(loc=mean_age, scale=std_age, size=n_samples)

            return pd.DataFrame({'is_male': is_male,
                                 'has_med_exp': has_med_exp,
                                 'age': age})

        self.toydata = pd.concat([
            generate_toydata(rate_male=.5, rate_med_exp=.8, mean_age=45,
                             std_age=12, n_samples=15),
            generate_toydata(rate_male=.7, rate_med_exp=.1, mean_age=22,
                             std_age=2, n_samples=35)]).reset_index(drop=True)

    def test(self):
        data = group_by_feature_balance(
            self.toydata.reset_index(),
            id_col='index', cat_col=['is_male', 'has_med_exp'], num_col='age',
            group_col='group', seed=0)
        cols = ['is_male', 'has_med_exp', 'age']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, var in zip(axes, cols):
            sns.violinplot(
                data=data, y=var, hue='group', split=True, ax=ax)
        fig.savefig('tests/output/test_group_by_feature_balance.png')
        data.astype(float).groupby('group').describe()

        MIN_PVAL = 0.05
        for var in cols:
            res = pg.ttest(data.query('group')[var], data.query('~group')[var])
            self.assertGreater(res['p-val'].iloc[0], MIN_PVAL)
