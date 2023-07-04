from typing import cast

import pandas as pd

from .misc import alphabet_to_num


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
