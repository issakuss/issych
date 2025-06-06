from typing import Sequence

import numpy as np
import pandas as pd

from .misc import alphabet2num


def loc_byalphabet(dataframe: pd.DataFrame, locs: str | Sequence[str]
                   ) -> pd.DataFrame:
    """
    :py:class:`pandas.DataFrame` から、アルファベットで指定した列を抽出します。

    Parameters
    ----------
    dataframe: :py:class:`pandas.DataFrame`
        このデータフレームから列を抽出します。
    locs: Sequence
        タプルやリストなどでアルファベットを指定します。

        Examples:

        >>> loc_byalphabet(df, ['A'])
        >>> loc_byalphabet(df, ['A', 'AC'])

    Notes
    -----
    MS Excelにおける列表示をそのまま利用しつつ列を指定できます。
    ただしPythonに読み込んだ際、一部列をIndexに指定するなどした際は注意してください。
    その場合、MS Excelに表示されたアルファベットが示す列と、抽出される列がズレます。
    27列目を指定する際には、MS Excelと同様 'AA' を使います。

    Tested with: :py:func:`monshi.Monshi.label`

    Examples
    --------
    >>> loc_byalphabet(dataframe, ['A', 'C'])
    >>> dataframe.iloc[:, [0, 2]]

    これらは同じ意味になります。

    See also
    --------
    loc_range_byalphabet: 範囲で指定するならこっち
    """
    index = [alphabet2num(loc) - 1 for loc in locs]
    return dataframe.iloc[:, index]


def loc_range_byalphabet(dataframe: pd.DataFrame, range_min: str,
                         range_max: str) -> pd.DataFrame:
    """
    :py:class:`pandas.DataFrame` から、アルファベットで指定した列範囲を抽出します。

    Parameters
    ----------
    dataframe: :py:class:`pandas.DataFrame`
        このデータフレームから列を抽出します。
    range_min: str
        ここで指定したアルファベット以降の列を抽出します。
    range_max: str
        ここで指定したアルファベット以前の列を抽出します。

        .. warning::

            抽出される列には、 `range_max` に指定された列も含まれます
            つまり、 `pandas.DataFrame().loc[]` による指定方法とは動作が異なります。

    Notes
    -----
    MS Excelにおける列表示をそのまま利用しつつ列範囲を指定できます。

    .. warning::

        ただしPythonに読み込んだ際、一部列をIndexに指定するなどした際は注意してください。
        その場合、MS Excelに表示されたアルファベットが示す列と、抽出される列がズレます。

    Tested with: :py:meth:`monshi.Monshi.label`

    Examples
    --------
    >>> loc_range_byalphabet(dataframe, 'A', 'C')
    >>> dataframe.iloc[:, 0:3]

    これらは同じ意味になります。

    See also
    --------
    loc_byalphabet: 列を一つずつ指定するならこっち
    """
    mini = alphabet2num(range_min) - 1
    maxi = alphabet2num(range_max)
    return dataframe.iloc[:, mini:maxi]


def loc_cols_name_startswith(dataframe: pd.DataFrame, head_colname: str
                             ) -> pd.DataFrame:
    """
    ``head_colname`` から始まる列名の列を抽出します。

    Parameters
    ----------
    dataframe: :py:class:`pandas.DataFrame`
        このデータフレームから列を抽出します。
    head_colname: str
        ここで指定した文字列から始まる列名のを抽出します。

    Notes
    -----
    Tested with: :py:meth:`monshi.Monshi.label`
    """
    are_target = [col.startswith(head_colname) for col in dataframe.columns]
    return dataframe.loc[:, are_target]


def vec2sqmatrix(vec: Sequence) -> np.ndarray:
    """
    Notes
    -----
    ベクトルを正方行列の :py:class:`numpy.ndarray` に変換します。

    Examples
    --------
    >>> vec2sqmatrix([1, 2, 3, 4])
    np.ndarray([[1, 2],
                [3, 4])
    """
    length = np.sqrt(len(vec))
    if not length.is_integer():
        raise RuntimeError(
            f'ベクトルの長さが{length}のため、正方行列に変換できません。')
    length = int(length)
    return np.array(vec).reshape(length, length)
