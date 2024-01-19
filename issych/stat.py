from typing import Optional, Sequence

import numpy as np
import pandas as pd


class Pvalue2SigMark:
    def __init__(self, thresholds: Optional[dict]=None, ns_comment: str=''):
        """
        thresholds: dict
            {p-value_threshold: 'significance_mark'}.
            Strictest threshold must be first.
        ns_comment: str
            Comment to be returned when non-significant.
        """
        DEFAULT = {0.01: '**', 0.05: '*', 0.10: '†'}
        self.thresholds = thresholds or DEFAULT
        self.ns_comment = ns_comment

    def __call__(self, pvalue: float):
        for thr, mark in self.thresholds.items():
            if pvalue < thr:
                return mark
        return self.ns_comment


def arcsine_sqrt(r: float) -> float:
    if np.isnan(r):
        return np.nan
    if not (0. <= r <= 1.):
        raise ValueError()
    return np.arcsin(np.sqrt(r))


def convert_to_iqrs(vec_origin: Sequence[float] | pd.Series) -> np.ndarray:
    """
    Rescale values relative to IQR.

    >> convert_to_iqrs([1, 2, 3, 4, 100])
    array([-0.5,  0. ,  0. ,  0.5,  24. ])
    """
    vec_origin = pd.Series(vec_origin)
    vec_dropna = pd.Series(vec_origin).dropna()
    vec_to_ret = vec_origin.copy()
    desc = vec_dropna.describe()
    low = desc['25%']
    high = desc['75%']
    iqr = high - low

    vec_to_ret[(low < vec_origin) & (vec_origin < high)] = 0.

    are_higher = vec_origin >= high
    are_lower = vec_origin <= low
    vec_to_ret[are_higher] = (vec_origin[are_higher] - high) / iqr
    vec_to_ret[are_lower] = (low - vec_origin[are_lower]) / iqr * -1
    return vec_to_ret.values


def nanzscore(vec: Sequence[float] | pd.Series) -> np.ndarray:
    """
    Tested in test_convert_to_iqrs.py
    """
    vec = np.array(vec)
    return (vec - np.nanmean(vec)) / np.nanstd(vec)


def nanzscore2value(zscore: float, vec: Sequence[float] | pd.Series) -> float:
    """
    Tested in test_convert_to_iqrs.py
    """
    vec = np.array(vec)
    return (zscore * np.nanstd(vec)) + np.nanmean(vec)