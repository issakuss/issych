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
