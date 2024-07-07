from typing import Any, Dict


class Dictm(Dict):
    """
    Dictm is almost equal to the built-in dict.
    However, it allows for calling contents using dot notation.

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