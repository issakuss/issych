from typing import Any, Callable, Dict, List
import contextlib
from functools import wraps
import time
from pathlib import Path
from copy import deepcopy

from ruamel.yaml import YAML
from ruamel.yaml.representer import Representer
from dynaconf.base import LazySettings
from dynaconf import Dynaconf
import joblib


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


class Int(int):
    """YAML出力時に !Int タグを付与するためのラッパークラス"""
    pass

class Float(float):
    """YAML出力時に !Float タグを付与するためのラッパークラス"""
    pass

class IntMean(float):
    """YAML出力時に !IntMean タグを付与するためのラッパークラス"""
    pass

class NGTO(float):
    """YAML出力時に !NGTO タグを付与するためのラッパークラス"""
    pass

class Pval(float):
    """YAML出力時に !Pval タグを付与するためのラッパークラス"""
    pass

class Str(str):
    """YAML出力時に !Str タグを付与するためのラッパークラス"""
    pass


class Dictm(Dict):
    """
    組み込みの辞書型の拡張版です。

    組み込みの辞書型と同じ機能を持っています。
    加えて、以下の例のように、ドットを使って要素を呼び出すことができます
    （プログラムをより簡易に書くための機能です）。
    また、組み込みの辞書型やPython-Box、Munchにはない、いくつかのメソッドを持っています。

    Examples
    --------
    >>> foo = Dictm({'a': 1, 'b': 2})
    >>> foo.a
    1
    """

    def __init__(self, *args, **kwargs):
        """
        ４パターンの作り方があります。

        Examples
        --------

        >>> # 1. 辞書を引数に取る方法
        >>> foo = Dictm({'a': 1, 'b': 2})
        >>> #
        >>> # 2. キーワードと引数を取る方法
        >>> foo = Dictm(a=1, b=2)
        >>> #
        >>> # 3. DynaconfのLazySettingsを引数に取る方法
        >>> foo = Dictm(Dynaconf(settings_files='settings.toml'))
        >>> #
        >>> # 4. Dynaconfに対応した設定ファイルのパスを引数に取る方法
        >>> foo = Dictm('settings.toml')
        """
        def _dictmize_nested(mydict: dict):
            return {k: Dictm(v) if isinstance(v, dict) else v
                    for k, v in mydict.items()}

        if len(args) == 1:
            match args[0]:
                case dict():
                    mydict = args[0]
                case LazySettings():
                    mydict = args[0].as_dict()
                    mydict = {k.lower(): v if isinstance(v, dict) else v
                              for k, v in mydict.items()}
                    args = (mydict,)
                case str() | Path() as mypath:
                    mypath = Path(mypath)
                    if not mypath.exists():
                        raise FileNotFoundError(
                            '以下のパスが指定されましたが、ファイルが存在しません:'
                            f'{mypath.resolve()}')
                    loaded_dynaconf = Dynaconf(settings_files=mypath)
                    _ = loaded_dynaconf  # to except TOMLDecodeError
                    mydict = Dictm(loaded_dynaconf)
        else:
            mydict = dict(*args, **kwargs)
        super().__init__(_dictmize_nested(mydict))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('__getstate__','__setstate__'):
            return super().__setattr__(name, value)
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.clear()
        self.update(state)

    def __deepcopy__(self, memo):
        return Dictm(deepcopy(dict(self), memo))

    def __or__(self, other):
        return Dictm(dict(self) | dict(other))

    def flatten(self):
        """
        ネストされた辞書を展開します。
        値が辞書の場合はその中身を取り出してマージし、辞書でない場合は元のキーと値のまま保持します。

        Examples
        --------
        >>> foo = Dictm({'A': {'a': 1, 'b': 2}, 'B': {'c': 3, 'd': 4}}) 
        >>> foo.flatten()
        {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        >>> bar = Dictm({'A': {'a': 1}, 'scalar': 2})
        >>> bar.flatten()
        {'a': 1, 'scalar': 2}
        """
        flattened = Dictm()
        for k, v in self.items():
            if isinstance(v, dict):
                flattened |= Dictm(v).flatten()
            else:
                flattened[k] = v
        return flattened

    def may(self, key: str) -> str:
        """
        辞書に該当するキーがあれば対応する要素を返します。
        該当するキーがなければ、キーをそのまま返します。
        `foo.get(key, key)` と同じですが、少しだけ簡潔に書けます。

        Examples
        --------
        >>> foo = Dictm({'a': 1, 'b': 2})
        >>> foo.may('a')
        1
        >>> foo.may('c')
        'c'
        """
        val = self.get(key)
        if val is None:
            return key
        return val

    def drop(self, key: str | List[str], skipnk: bool=False):
        """
        指定したキーと、それに対応する値のペアを削除した Dictm を返します。
        該当するキーがないとエラーになりますが、skipnk が True のときは無視されます。

        Parameters
        ----------
        key : str or list of str
            削除するキー。
        skipnk : bool
            Skip No Key。
            True のとき、該当するキーがなくてもエラー返しません。

        Examples
        --------
        >>> dictm = Dictm({'a': 1, 'b': 2, 'c': 3, 'd': 4})
        >>> dictm.drop('a').keys()
        ['b', 'c', 'd']
        >>> dictm.drop(['a', 'b']).keys()
        ['c', 'd']
        >>> dictm.drop(['d', 'e'], skip_nk=True).keys()
        ['a', 'b', 'c']
        """
        dropkeys = [key] if isinstance(key, str) else list(key)
        nokeys = set(dropkeys) - set(self.keys())
        if (not skipnk) and nokeys:
            raise RuntimeError(f'{nokeys} がキーにありません。'
                               'skipnk=True にするとこのエラーを抑制できます。')
        return Dictm({key: val
                      for key, val in self.items() if key not in dropkeys})

    def to_yaml(self, out_path_yaml: str | Path):
        """
        Yamlファイルとして出力します。
        値がカスタムクラスの場合は、クラス名がカスタムタグとして付与されます。

        Parameters
        ----------
        out_path_yaml : str or Path
            出力先のファイルパス。
        """

        yaml = YAML()
        yaml.default_flow_style = False

        def represent_custom(representer, data):
            tag = f"!{data.__class__.__name__}"
            return representer.represent_scalar(tag, str(data))

        yaml.representer.add_representer(Dictm, Representer.represent_dict)
        yaml.representer.add_multi_representer(object, represent_custom)

        with open(out_path_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(self, f)
