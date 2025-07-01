from typing import Any, Dict, List
import string
from pathlib import Path

from dynaconf.base import LazySettings
from dynaconf import Dynaconf


class Dictm(Dict):
    def __init__(self, *args, **kwargs):
        """
        組み込みの辞書型と同じ機能を持っています。
        組み込みの辞書型にはない、いくつかのメソッドを持っています。
        加えて、以下の例のように、ドットを使って要素を呼び出すことができます
        （プログラムをより簡易に書くための機能です）。

        >>> foo = Dictm({'a': 1, 'b': 2})
        >>> foo.a
        1

        ４パターンの作り方があります。

        - 1. 辞書を引数に取る方法

        >>> foo = Dictm({'a': 1, 'b': 2})

        - 2. キーワードと引数を取る方法

        >>> foo = Dictm(a=1, b=2)

        - 3. DynaconfのLazySettingsを引数に取る方法

        >>> foo = Dictm(Dynaconf(settings_files='settings.toml'))

        - 4. Dynaconfに対応した設定ファイルのパスを引数に取る方法

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
                    print(loaded_dynaconf)  # to except TOMLDecodeError
                    mydict = Dictm(loaded_dynaconf)
        else:
            mydict = dict(*args, **kwargs)
        super().__init__(_dictmize_nested(mydict))
        self.__dict__ = self

    def __getattr__(self, _: str) -> Any: ...

    def __or__(self, other):
        return Dictm(dict(self) | dict(other))

    def flatten(self):
        """
        ネストされた辞書を開きます。

        Examples
        --------
        >>> foo = Dictm({'A': {'a': 1, 'b': 2}, 'B': {'c': 3, 'd': 4}}) 
        >>> foo.flatten()
        {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        """
        flatten = Dictm()
        for v in self.values():
            flatten |= v
        return flatten

    def may(self, key: str) -> str:
        """
        辞書に該当するキーがあれば対応する要素を返します。
        該当するキーがなければ、キーをそのまま返します。

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
        該当するキーがないとエラーになりますが、skip_nk が True のときは無視されます。

        Parameters
        ----------
        key : str or list of str
            削除するキー。
        skipnk : bool
            skip No Key。
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
        return Dictm({key: self[key]
                      for key in self.keys() if key not in dropkeys})


class Pathm:
    def __init__(self, template: str='.'):
        """
        組み込みのpathlib.Pathとほぼ同じ機能を持っています。
        ただし、テンプレートを用いた流動的なパスの指定機能が加わっています。

        Parameters
        ----------
        template : str
            パスのテンプレートとなる文字列。
            {key} 形式で変数を指定します。
            例：'{parent}/interim/{sub_id}/data.csv'

        Examples
        --------
        >>> foo = Pathm('{parent}/interim/{sub_id}/data.csv')
        >>> foo(parent='data', sub_id='s001')
        PosixPath('data/interim/s001/data.csv')
        >>> foo(parent='data', sub_id'='s002')
        PosixPath('data/interim/s002/data.csv')

        >>> bar = Pathm('settings.toml')
        >>> bar.exists()
        True
        """
        self.template = template
        self._last_path = None if self._has_var(template) else Path(template)

    def __call__(self, **kwargs):
        formatted = self.template.format(**kwargs)
        self._last_path = Path(formatted)
        return self._last_path

    def __getattr__(self, attr):
        if self._last_path is None:
            raise AttributeError('パスの変数に指定がありません')
        return getattr(self._last_path, attr)

    def __str__(self):
        return self.template

    def __truediv__(self, other):
        new_template = f'{self.template.rstrip("/")}/{str(other).lstrip("/")}'
        return Pathm(new_template)

    def __fspath__(self):
        if self._last_path is None:
            return self.template
        return str(self._last_path)

    def _has_var(self, template):
        return any(field_name for _, field_name, _, _
                   in string.Formatter().parse(template) if field_name)
