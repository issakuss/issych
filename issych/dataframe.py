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
    