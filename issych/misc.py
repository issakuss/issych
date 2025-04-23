from typing import Callable, List
import contextlib
from functools import wraps
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler



def alphabet2num(alphabet: str) -> int:
    """
    アルファベットを数値に変換します。

    Examples
    --------
    >>> alphabet2num('A')
    1
    >>> alphabet2num('a')
    1
    >>> alphabet2num('Z')
    26
    >>> alphabet2num('AA')
    27
    """
    N_ALPHABET = 26
    num = 0
    alphabet = alphabet.upper()
    for i, item in enumerate(alphabet):
        order = ord(item) - ord('A') + 1
        num += order * pow(N_ALPHABET, len(alphabet) - i - 1)
    return num


def meas_exectime(func: Callable):
    """
    デコレータで関数の実行時間を測定し、表示します。

    Examples
    --------
    >>> @meas_exectime
    >>> def foo():
    >>>     time.sleep(1)
    >>> foo()
    1.000 sec.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        init = time.time()
        returnval = func(*args, **kwargs)
        elapsed = time.time() - init
        print(f'{func.__name__}: {elapsed:.3f} sec.')
        return returnval
    return wrapper


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    並列処理を行なっている際のプログレスバーを表示します。

    Examples
    --------
    >>> from joblib import Parallel, delayed
    >>> from tqdm import tqdm
    >>>
    >>> def process_item(item):
    >>>     time.sleep(0.1)
    >>>     return item * 2
    >>> 
    >>> items = list(range(10))
    >>> paralleled = (delayed(process_item)(item) for item in items)
    >>> with tqdm_joblib(tqdm(total=len(items))) as pbar:
    >>>     results = Parallel(n_jobs=-1)(paralleled)

    Reference
    -------
    https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    こちらに投稿されたコードをそのまま使っています。
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def group_by_feature_balance(df: pd.DataFrame, id_col: str,
                             cat_col: str | List[str], num_col: str,
                             group_col: str, seed: int=0) -> pd.DataFrame:
    """
    年齢や性別などの特徴に基づいて、なるべく均等にグループ分けします。

    Parameters
    ----------
    df : pd.DataFrame
        グループ分けに用いる特徴量を含むデータフレーム。
    id_col : str
        サンプルのIDを示す列名。
    cat_col : str or list of str
        グループ分けに用いるカテゴリ変数。
    num_col : str
        グループ分けに用いる数値変数。
        一つしか指定できません。
    group_col : str
        グループ分けの結果を格納する列の名前。
        元のデータフレームにない列名の場合、新規列が作成されます。
    seed: int, default 0
        乱数のシード。

    Returns
    -------
    grouped_df : pd.DataFrame
        グループ分けの結果がgroup_colに格納されたデータフレーム。

    Notes
    -----
    R の groupdata2 パッケージを使用しています。
    事前に R へこのパッケージをインストールしてください。

    References
    ----------
    https://cran.r-project.org/web/packages/groupdata2/index.html

    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    df[id_col] = df[id_col].astype(str)
    if isinstance(cat_col, str):
        cat_col = [cat_col]

    ro.r.assign('id_col', id_col)
    ro.r.assign('cat_col', ro.StrVector(cat_col))
    ro.r.assign('num_col', num_col)
    ro.r.assign('seed', seed)

    with (ro.default_converter + pandas2ri.converter).context():
        ro.r.assign('data', ro.conversion.get_conversion().py2rpy(df))
    ro.r('''
        library("groupdata2")
        set.seed(seed)
        data[[id_col]] <- as.factor(data[[id_col]])
        partitioned <- partition(
            data = data,
            p = 0.5,
            cat_col = cat_col,
            num_col = num_col,
            id_col = id_col)
         ''')
    group_a = pandas2ri.rpy2py(ro.r('partitioned[[1]]'))
    group_b = pandas2ri.rpy2py(ro.r('partitioned[[2]]'))
    group_a[group_col] = True
    group_b[group_col] = False
    return pd.concat([group_a, group_b])
