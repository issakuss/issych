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

    def __getattr__(self, name: str) -> Any: ...

    def __or__(self, other):
        return Dictm(dict(self) | dict(other))


if __name__ == '__main__':
    dictm = Dictm({'a': 1, 'b': 2})
    dictm_to_add = Dictm({'c': 3})
    dictm = dictm | dictm_to_add
    assert dictm.a == 1 and dictm.b == 2 and dictm.c == 3
    print('Dictm is OK')
