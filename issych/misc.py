from typing import Sequence
from functools import wraps
import time

import numpy as np


def alphabet_to_num(alphabet: str):
    """
    >> alphabet_to_num('A')
    1
    >> alphabet_to_num('a')
    1
    >> alphabet_to_num('Z')
    26
    >> alphabet_to_num('AA')
    27
    """
    N_ALPHABET = 26
    num = 0
    alphabet = alphabet.upper()
    for i, item in enumerate(alphabet):
        order = (ord(item) - ord('A') + 1)
        num += order * pow(N_ALPHABET, len(alphabet) - i - 1)
    return num


def vec2sqmatrix(vec: Sequence) -> np.ndarray:
    """
    >> vec2sqmatrix([1, 2, 3, 4])
    np.ndarray([[1, 2],
                [3, 4])
    """
    length = np.sqrt(len(vec))
    if not length.is_integer():
        raise ValueError('This length cannot convert to square matrix')
    length = int(length)
    return np.array(vec).reshape(length, length)


def meas_exectime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        init = time.time()
        returnval = func(*args, **kwargs)
        elapsed = time.time() - init
        print(f'{func.__name__}: {elapsed:.3f} sec.')
        return returnval
    return wrapper
