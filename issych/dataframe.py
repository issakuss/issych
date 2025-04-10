from typing import Sequence

import numpy as np
import pandas as pd

from .misc import alphabet2num


def loc_byalphabet(dataframe: pd.DataFrame, locs: Sequence[str]
                   ) -> pd.DataFrame:
    """
    pd.DataFrameから、アルファベットで指定した列を抽出します。
    MS Excelにおける列表示をそのまま利用しつつ列を指定できます。
    ただしPythonに読み込んだ際、一部列をIndexに指定するなどした際は注意してください。
    その場合、MS Excelに表示されたアルファベットが示す列と、抽出される列がズレます。

    >> loc_byalphabet(dataframe, ['A', 'C'])
    は、
    >> dataframe.iloc[:, [0, 2]]
    と同等になります。

    Tested with: monshi.Monshi().label()
    """
    index = [alphabet2num(loc) - 1 for loc in locs]
    return dataframe.iloc[:, index]


def loc_range_byalphabet(dataframe: pd.DataFrame, range_min: str,
                         range_max: str) -> pd.DataFrame:
    """
    pd.DataFrameから、アルファベットで指定した列範囲を抽出します。
    抽出される列には、range_min, range_maxに指定された列も含まれます
    （つまり、pd.DataFrame().loc[]による指定方法とは動作が異なります）。
    MS Excelにおける列表示をそのまま利用しつつ列範囲を指定できます。
    ただしPythonに読み込んだ際、一部列をIndexに指定するなどした際は注意してください。
    その場合、MS Excelに表示されたアルファベットが示す列と、抽出される列がズレます。

    >> loc_range_byalphabet(dataframe, 'A', 'C')
    は、
    >> dataframe.iloc[:, 0:3]
    と同等になります。

    Tested with: monshi.Monshi().label()
    """
    mini = alphabet2num(range_min) - 1
    maxi = alphabet2num(range_max)
    return dataframe.iloc[:, mini:maxi]


def loc_cols_name_startswith(dataframe: pd.DataFrame, head_colname: str
                             ) -> pd.DataFrame:
    """
    head_colnameから始まる列名の列を抽出します。

    Tested with: monshi.Monshi().label()
    """
    are_target = [col.startswith(head_colname) for col in dataframe.columns]
    return dataframe.loc[:, are_target]


def vec2sqmatrix(vec: Sequence) -> np.ndarray:
    """
    ベクトルを正方行列のnp.ndarrayに変換します。
    >> vec2sqmatrix([1, 2, 3, 4])
    np.ndarray([[1, 2],
                [3, 4])
    """
    length = np.sqrt(len(vec))
    if not length.is_integer():
        raise RuntimeError(
            f'ベクトルの長さが{length}のため、正方行列に変換できません。')
    length = int(length)
    return np.array(vec).reshape(length, length)
