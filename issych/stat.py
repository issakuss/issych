from typing import Optional, Sequence, Dict
from math import floor

import numpy as np
import pandas as pd

from .typealias import Number, Vector


class Pvalue2SigMark:
    """
    与えられたp値に基づいて、有意性の程度を表すマークを返します。

    Tested with: figure.SigMarker()
    """
    def __init__(self, thresholds: Optional[Dict[float, str]]=None):  # Default {} is dangerous
        """
        thresholds: dict
            p値の閾値と、その閾値を下回った際に返すべき文字列を示す辞書。
            >> {0.01: '**', 0.05: '*', 0.10: '†'}
            いずれの閾値も下回らない場合のマークを指定したい場合は、
            >> {0.05: '*', 1.1: 'n.s.'}
            としてください。
            そういった指定がない場合は、''が返ります。
        """
        DEFAULT = {0.01: '**', 0.05: '*', 0.10: '†'}
        thresholds = thresholds or DEFAULT
        self.thresholds = dict(sorted(thresholds.items()))

    def __call__(self, pvalue: float) -> str:
        for thr, mark in self.thresholds.items():
            if pvalue < thr:
                return mark
        return ''


def arcsine_sqrt(value: Number) -> float:
    """
    アークサイン平方根変換を行います。
    0〜1をとる比率データの分散に対して、正規性を向上させる可能性があります。

    Tested with: test_calculation.py
    """
    if np.isnan(value):
        return np.nan
    if not 0. <= value <= 1.:
        raise ValueError('与えられた値{value}は0〜1の範囲にありません。')
    return np.arcsin(np.sqrt(value))


def fisher_z(r: Number) -> float:
    """
    FisherのZ変換を行います。
    相関係数など、-1〜1の範囲にあるデータに対して、正規性を向上させる可能性があります。
    """
    if r  == 1:
        return float('inf')
    if r == -1:
        return float('-inf')
    if not -1. <= r <= 1.:
        raise ValueError('与えられた値{r}は-1〜1の範囲にありません。')
    return 0.5 * np.log((1 + r) / (1 - r))


def convert_to_iqrs(vec_origin: Vector) -> np.ndarray:
    """
    数値のベクトルをIQRに基づいてスケーリングします。

    >> convert_to_iqrs([1, 2, 3, 4, 100])
    array([-0.5,  0. ,  0. ,  0.5,  24. ])

    Tested with: test_iqr.py
    """
    vec_origin = pd.Series(vec_origin).astype(float)
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


def _prep_nanz(vec: Vector) -> np.ndarray:
    if isinstance(vec, pd.DataFrame):
        raise ValueError('pd.DataFrameはサポートされていません。')
    if isinstance(vec, pd.Series):
        vec = vec.astype(float)
    vec = np.array(vec)
    return vec


def nanzscore(vec: Vector) -> np.ndarray:
    """
    NaNを含むベクトルに対して、NaNを無視してZスコア変換を行います。

    Tested with: test_convert_to_iqrs.py
    """
    vec = _prep_nanz(vec)
    return (vec - np.nanmean(vec)) / np.nanstd(vec)


def nanzscore2value(zscore: Number, vec: Vector) -> float:
    """
    指定したZスコアを、指定したベクトルにおける元の値に変換します。
    ベクトルにおいてNaNは無視されます。

    >>> vec = np.array([1, 2, 3, 4, 5, np.nan])
    >>> zscore = 0.5
    >>> nanzscore2value(zscore, vec)
    3.0

    Tested in test_convert_to_iqrs.py
    """
    vec = _prep_nanz(vec)
    return (zscore * np.nanstd(vec)) + np.nanmean(vec)


def value2nanzscore(value: float, vec: Sequence[float] | pd.Series) -> float:
    """
    指定したベクトルにおける、指定した値のZスコアを計算します。
    ベクトルにおいてNaNは無視されます。

    >>> vec = np.array([1, 2, 3, 4, 5, np.nan])
    >>> value = 3
    >>> value2nanzscore(value, vec)
    0.0

    Tested in test_convert_to_iqrs.py
    """
    vec = _prep_nanz(vec)
    return (value - np.nanmean(vec)) / np.nanstd(vec)
