import unittest

import seaborn as sns

from issych.figure import prepare_ax, mask_area


class TestMaskArea(unittest.TestCase):
    def test(self):
        data = sns.load_dataset('fmri')
        fig, ax = prepare_ax(path_ini='tests/testdata/config/rcparams.ini')
        sns.lineplot(x="timepoint", y="signal", data=data, ax=ax)
        mask_area(3.5, 7.5, orient='h', color='gray', ax=ax, zorder=-1)
        mask_area(12.5, 12.5, orient='h', ax=ax, ec='white')
        mask_area(0.08, 0.12, orient='vert', ax=ax, color='red')
        fig.savefig('test.png')
