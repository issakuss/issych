from typing import Any, Dict


class Dictm(Dict):
    """
    組み込みの辞書型と同じ機能を持っています。
    組み込みの辞書型にはない、いくつかのメソッドを持っています。
    加えて、以下の例のように、ドットを使って要素を呼び出すことができます
    （プログラムをより簡易に書くための機能です）。

    >> foo = Dictm({'a': 1, 'b': 2})
    >> foo.a
    1

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, _: str) -> Any: ...

    def __or__(self, other):
        return Dictm(dict(self) | dict(other))

    def flatten(self):
        """
        ネストされた辞書を開きます。
        >> foo = Dictm({'A': {'a': 1, 'b': 2}, 'B': {'c': 3, 'd': 4}}) 
        >> foo.flatten()
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
        >> foo = Dictm({'a': 1, 'b': 2})
        >> foo.may('a')
        1
        >> foo.may('c')
        'c'
        """
        val = self.get(key)
        if val is None:
            return key
        return val

    def full(self, *args) -> str:
        """
        Dictm.may()と同じ機能を持ちます。
        主に、単語の省略形を完全形に直すために用います。
        >> foo = Dictm({'bdi2': 'BDI-II'})
        >> foo.full('bdi2')
        'BDI-II'
        >> foo.full('cesd')
        'cesd'
        """
        return self.may(*args)