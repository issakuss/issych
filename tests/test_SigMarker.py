import unittest

import seaborn as sns

from issych.stat import Pvalue2SigMark

from issych.figure import prepare_axes, SigMarker


class TestSigMarker(unittest.TestCase):
    def test(self):
        data = sns.load_dataset('tips')
        pvalues_btw_day = [[0, 1, 0.2],  # x, y, pvalue; fictious value.
                           [0, 2, 0.2],
                           [0, 3, 0.2],
                           [1, 2, 0.2],
                           [1, 3, 0.06],
                           [2, 3, 0.2]]
        pvalues_btw_hue = [[0, 1, 0.2], 
                           [2, 3, 0.011],
                           [4, 5, 0.2],
                           [6, 7, 0.2]]

        fig, axes = prepare_axes(n_col=2, axsizeratio=(.5, 1.),
                                 path_ini='tests/testdata/config/rcparams.ini')

        for ax, ns, maxthr in zip(axes, ['n.s.', ''], [1.0, 0.1]):
            sns.barplot(
                data=data, x='day', y='total_bill', hue='sex', ax=ax)
            p2sig = Pvalue2SigMark(ns_comment=ns)
            marker = SigMarker(ax)
            for x, y, p in pvalues_btw_day:
                if p < maxthr:
                    marker.mark(
                        'xticks', pos_from=x, pos_to=y, comment=p2sig(p))
            for x, y, p in pvalues_btw_hue:
                if p < maxthr:
                    marker.mark(
                        'patches', pos_from=x, pos_to=y, comment=p2sig(p))
            ax.legend_.remove()
