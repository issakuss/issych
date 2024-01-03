from typing import cast, Any, Optional, List

import pandas as pd

from .misc import alphabet_to_num
from .fileio import load_config
from .dataclass import Dictm


def extract_byalphabet(dataframe: pd.DataFrame, range_min: str, range_max: str
    ) -> pd.DataFrame:
    """
    extract_byalphabet(dataframe, 'A', 'C)
    is equall to dataframe.iloc[:, 0:3]
    """
    mini = alphabet_to_num(range_min) - 1
    maxi = alphabet_to_num(range_max)
    return dataframe.iloc[:, mini:maxi]


def extract_colname_startswith(dataframe: pd.DataFrame, head_colname: str):
    """
    Extract columns whose name starts with head_colname.
    """
    are_target = [col.startswith(head_colname) for col in dataframe.columns]
    return dataframe.loc[:, are_target]


def fullform_index(dataframe: pd.DataFrame, abbr: dict,
                   for_index: bool=True, for_columns: bool=True
                   ) -> pd.DataFrame:

    def make_lowercase(v: Any) -> Any:
        if isinstance(v, str):
            return v.lower()
        return v

    def eachind(ind: pd.Index, abbr: dict) -> pd.Index:
        if isinstance(ind, pd.MultiIndex):
            ind = pd.MultiIndex.from_frame(
                (ind.to_frame(allow_duplicates=True).replace(abbr)))
            ind.names = (None,) * ind.nlevels  # type: ignore
            return ind
        old_ind = ind.to_frame().map(make_lowercase)
        ind = pd.Index(old_ind.replace(abbr).iloc[:, 0])
        ind.name = None
        return ind
        
    dataframe = dataframe.copy()

    index, columns = [eachind(ind, abbr)
                      for ind in [dataframe.index, dataframe.columns]]
    if for_index: dataframe.index = index
    if for_columns: dataframe.columns = columns
    
    return dataframe


def pad_zero(cell: float | int | str, sdgt: Optional[int]=None) -> str:
    if isinstance(cell, int):
        return str(cell)
    if pd.isna(cell):
        return ''
    sdgt = sdgt or load_config('config/misc.ini').general.sdgt
    if cell == '-':
        return str(cell)
    cell = str(cell)
    n_shortage = sdgt - (len(cell) - cell.find('.') - 1)
    return cell + ('0' * n_shortage)


def set_sdgt(dataframe: pd.DataFrame, sdgts: dict,
             cols_rate: List[str]=[], cols_p: List[str]=[],
             thr_p: Optional[float]=None) -> pd.DataFrame:
    """
    sdgt must include 'main', 'pvalue' keys.
    """
    def eachcol(data: pd.Series, sdgts: dict,
                cols_rate: List[str], cols_p: List[str], thr_p: Optional[float]
                ) -> pd.Series:
        sdgts = Dictm(sdgts)
        if data.name in cols_p:
            data_ = data.copy().round(sdgts.pvalue)
            data_ = data_.astype(object).fillna('').astype(str)
            if thr_p:
                data_.loc[data < thr_p] = f'< {str(thr_p).replace("0.", ".")}'
            return data_.astype(str).apply(lambda x: x.replace('0.', '.'))

        data = data.fillna(float('nan')).round(sdgts.main)
        data = data.apply(pad_zero, args=(sdgts.main,))

        if data.name in cols_rate:
            data = data.apply(lambda x: x.replace('0.', '.'))

        return data

    return dataframe.apply(eachcol, args=(sdgts, cols_rate, cols_p, thr_p))



def flatten_multi_index(dataframe, pad='&emsp;', n_pad=4):
    """
    space code = '&emsp;'
    """
    def insert_pad_to_index(dataframe, pad, n_pad):
        if not isinstance(dataframe.index, pd.MultiIndex):
            dataframe.index = [(pad * n_pad) + index
                                for index in dataframe.index]
            return
        newindex = []
        for row in dataframe.index:
            row = cast(pd.Series, row)
            newindex.append(tuple((pad * n_pad) + row[i]
                                    for i, _ in enumerate(row)))
        dataframe.index = pd.MultiIndex.from_frame(pd.DataFrame(newindex))
        dataframe.index.names = [None] * len(dataframe.index.names)  # type: ignore

    def insert_empty_row(dataframe, name):
        dataframe_ = dataframe.T
        dataframe_.insert(0, name, pd.NA)
        return dataframe_.T

    def eachindex(dataframe, index, pad, n_pad):
        dataframe = dataframe.loc[index, :]
        if (len(dataframe.index) == 1) and pd.isna(dataframe.index.name):
            dataframe.index = pd.Index([index])
            return dataframe

        dataframe = flatten_multi_index(dataframe, pad, n_pad)
        dataframe.index = dataframe.index.fillna('')
        insert_pad_to_index(dataframe, pad, n_pad)
        dataframe = insert_empty_row(dataframe, index)
        return dataframe

    if not isinstance(dataframe.index, pd.MultiIndex):
        return dataframe
    higher_index = list(dataframe.index.get_level_values(0))
    dfs = [eachindex(dataframe, index, pad, n_pad)
           for index in sorted(set(higher_index), key=higher_index.index)]
    return pd.concat(dfs)
