from typing import Optional, Sequence, Any, Mapping
from math import floor

import numpy as np
import pandas as pd

import numpy.typing as npt


class Pvalue2SigMark:
    """
    与えられたp値に基づいて、有意性の程度を表すマークを返します。
    """
    def __init__(self, thresholds: Optional[Mapping[float, str]]=None):
        """
        Parameters
        ----------
        thresholds: dict, optional
            p値の閾値と、その閾値を下回った際に返すべき文字列を示す辞書です。
            
            **Example**

            >>> {0.01: '**', 0.05: '*', 0.10: '†'}

            いずれの閾値も下回らない場合のマークを指定したい場合は、

            >>> {0.05: '*', 1.1: 'n.s.'}
            としてください。
            そういった指定がない場合は、''が返ります。
        """
        DEFAULT = {0.01: '**', 0.05: '*', 0.10: '†'}
        if thresholds is None:  # skip thresholds is {}
            thresholds = DEFAULT
        self.thresholds = dict(sorted(thresholds.items()))

    def __call__(self, pvalue: float) -> str:
        for thr, mark in self.thresholds.items():
            if pvalue < thr:
                return mark
        return ''


def arcsine_sqrt(value: float) -> float:
    """
    アークサイン平方根変換を行います。

    Notes
    -----
    一般に、0〜1をとる比率データの分散に対して、正規性を向上させる目的で使用されます。
    """
    if np.isnan(value):
        return np.nan
    if not 0. <= value <= 1.:
        raise ValueError('与えられた値{value}は0〜1の範囲にありません。')
    return np.arcsin(np.sqrt(value))





def convert_to_iqrs(vec_origin: npt.ArrayLike) -> np.ndarray:
    """
    数値のベクトルを IQR に基づいてスケーリングします。

    Examples
    --------
    >>> convert_to_iqrs([1, 2, 3, 4, 100])
    array([-0.5,  0. ,  0. ,  0.5,  24. ])
    """
    vec_origin = np.asarray(vec_origin, dtype=float)
    vec_to_ret = vec_origin.copy()
    
    low, high = np.nanpercentile(vec_origin, [25, 75])
    iqr = high - low

    if iqr == 0:
        vec_to_ret[:] = 0.
        return vec_to_ret

    vec_to_ret[(low < vec_origin) & (vec_origin < high)] = 0.

    are_higher = vec_origin >= high
    are_lower = vec_origin <= low
    vec_to_ret[are_higher] = (vec_origin[are_higher] - high) / iqr
    vec_to_ret[are_lower] = (vec_origin[are_lower] - low) / iqr

    return vec_to_ret


def _prep_nanz(vec: npt.ArrayLike) -> np.ndarray:
    if isinstance(vec, pd.DataFrame):
        raise ValueError('pd.DataFrameはサポートされていません。')
    if isinstance(vec, pd.Series):
        vec = vec.values
    return pd.to_numeric(vec, errors='coerce')





def nanzscore2value(zscore: float, vec: npt.ArrayLike) -> float:
    """
    指定したZスコアを、指定したベクトルにおける元の値に変換します。
    ベクトルにおいて NaN は無視されます。

    Examples
    --------
    >>> vec = np.array([1, 2, 3, 4, 5, np.nan])
    >>> zscore = 0.5
    >>> nanzscore2value(zscore, vec)
    3.707
    """
    vec = _prep_nanz(vec)
    return float((zscore * np.nanstd(vec)) + np.nanmean(vec))


def iqr2value(iqr: float, vec: npt.ArrayLike) -> float:
    """
    指定したIQRを、指定したベクトルにおける元の値に変換します。

    Examples
    --------
    >>> vec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> iqr2value(-0.5, vec)
    1.0
    >>> iqr2value(0.25, vec)
    8.0
    >>> iqr2value(1.5, vec)
    13.0
    """
    vec = np.asarray(vec, dtype=float)
    low, mid, high = np.nanpercentile(vec, [25, 50, 75])
    if iqr > 0.:
        return float(high + ((high - low) * iqr))
    if iqr < 0.:
        return float(low + ((high - low) * iqr))
    return float(mid)


def value2nanzscore(value: float, vec: Sequence[float] | pd.Series) -> float:
    """
    指定したベクトルにおける、指定した値のZスコアを計算します。
    ベクトルにおいてNaNは無視されます。

    Examples
    --------
    >>> vec = np.array([1, 2, 3, 4, 5, np.nan])
    >>> value = 3
    >>> value2nanzscore(value, vec)
    0.0
    """
    vec = _prep_nanz(vec)
    return float((value - np.nanmean(vec)) / np.nanstd(vec))


def kwargs4r(kwargs: Mapping[str, Any]) -> str:
    """
    キーワード引数をRのキーワード引数形式に変換します。
    R 内の変数を指定する場合は頭に @ をつけてください。

    Parameters
    ----------
    kwargs : dict
        キーワード引数の辞書。

    Returns
    -------
    str
        Rのキーワード引数形式の文字列。

    Examples
    --------
    >>> r('''
    >>>    myfunc <- function(val1, val2, comment) {
    >>>        paste0(comment, ": ", val1 + val2) }
    >>>    val1 <- 1
    >>> ''')
    >>> kwargs = {'val1': '@val1', 'val2': 2, 'comment': 'test'}
    >>> r(f'myfunc({kwargs4r(kwargs)})')    
    test: 3
    """
    rkwargs = ''
    for k, v in kwargs.items():
        if isinstance(v, (int, float)):
            rkwargs += f'{k} = {v}, '
        elif isinstance(v, str):
            if v.startswith('@'):
                rkwargs += f'{k} = {v[1:]}, '
            else:
                rkwargs += f'{k} = "{v}", '
        elif isinstance(v, Sequence):
            v = [f'"{v_}"' if isinstance(v_, str) else v_ for v_ in v]
            rkwargs += f'{k} = c({", ".join(map(str, v))}), '
        elif isinstance(v, dict):
            v = [f'"{k_}" = {v_}' for k_, v_ in v.items()]
            rkwargs += f'{k} = c({", ".join(v)}), '
        else:
            raise TypeError(f'引数の値は文字列、数値、またはそれらから成る Sequence でなければなりません: {type(v)}')
    return rkwargs.rstrip(', ')
