from typing import Sequence, Mapping

import numpy as np
import pandas as pd

from .misc import alphabet2num, IssychQuartoTag


def loc_byalphabet(dataframe: pd.DataFrame, locs: str | Sequence[str]
                   ) -> pd.DataFrame:
    """
    :py:class:`pandas.DataFrame` から、アルファベットで指定した列を抽出します。

    MS Excelにおける列表示をそのまま利用しつつ列を指定できます。
    27列目を指定する際には、MS Excelと同様 'AA' を使います。

    .. warning::

        Pythonに読み込んだ際、一部列をIndexに指定するなどした際は注意してください。
        その場合、MS Excelに表示されたアルファベットが示す列と、抽出される列がズレます。

    Parameters
    ----------
    dataframe: :py:class:`pandas.DataFrame`
        このデータフレームから列を抽出します。
    locs: Sequence
        タプルやリストなどでアルファベットを指定します。

    Examples
    --------
    >>> loc_byalphabet(dataframe, ['A', 'C', 'AC'])
    >>> dataframe.iloc[:, [0, 2, 29]]

    これらは同じ意味になります。

    See also
    --------
    loc_range_byalphabet: 範囲で指定するならこっち
    """
    if isinstance(locs, str):
        locs = [locs]
    index = [alphabet2num(loc) - 1 for loc in locs]
    return dataframe.iloc[:, index]


def loc_range_byalphabet(dataframe: pd.DataFrame, range_min: str,
                         range_max: str) -> pd.DataFrame:
    """
    :py:class:`pandas.DataFrame` から、アルファベットで指定した列範囲を抽出します。

    MS Excelにおける列表示をそのまま利用しつつ列範囲を指定できます。

    .. warning::

        ただしPythonに読み込んだ際、一部列をIndexに指定するなどした際は注意してください。
        その場合、MS Excelに表示されたアルファベットが示す列と、抽出される列がズレます。

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


def cols_startswith(dataframe: pd.DataFrame, head_colname: str
                    ) -> pd.DataFrame:

    """
    ``head_colname`` から始まる列名の列を抽出します。

    Parameters
    ----------
    dataframe: :py:class:`pandas.DataFrame`
        このデータフレームから列を抽出します。
    head_colname: str
        ここで指定した文字列から始まる列名の列を抽出します。
    """
    return dataframe.loc[:, dataframe.columns.str.startswith(head_colname)]


def vec2sqmatrix(vec: Sequence) -> np.ndarray:
    """
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


def add_tag(dataframe: pd.DataFrame, tags: Mapping[str, type[IssychQuartoTag]]
            ) -> pd.DataFrame:
    """
    `issych-quarto-template` で利用できる、タグ付きデータフレームを出力します。

    Parameters
    ----------
    dataframe: :py:class:`pandas.DataFrame`
        このデータフレームを出力します。
    tags: Mapping[str, type[IssychQuartoTag]]
        キーには `dataframe` の列名を指定します。
        値には表示したい、`issych-quarto-template` 用の型を指定します。
        
    Examples
    --------
    >>> dataframe.columns
    Index(['col1', 'col2', 'col3'], dtype='object')
    >>> tags = {'col1': Int, 'col2': Float, 'col3': IntMean}
    >>> add_tag(dataframe, tags)
    col1!Int, col2!Float, col3!IntMean
    ---
    1, 1.023, 1.2
    2, 2.034, 2.4
    ... 
    """

    dataframe_ = dataframe.copy()
    new_cols = {col: f'{col}!{tag.__name__}'
                for col, tag in tags.items() if col in dataframe_.columns}
    return dataframe_.rename(columns=new_cols)
