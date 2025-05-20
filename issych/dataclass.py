from typing import Any, Dict
from pathlib import Path

from dynaconf.base import LazySettings
from dynaconf import Dynaconf


class Dictm(Dict):
    def __init__(self, *args, **kwargs):
        """
        Notes
        -----
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
        Notes
        -----
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
        Notes
        -----
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
