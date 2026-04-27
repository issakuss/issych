from typing import Dict
import configparser

from pathlib import Path


def save_ini(data: Dict[str, Dict], out_path_ini: str | Path):
    """
    辞書を値に持つ、ネストされた辞書をINIファイルに保存します。

    Parameters
    ----------
    data: dict
        INIファイルに保存する辞書です。
    out_path_ini: str | Path
        INIファイルの保存先となるパスです。
    """

    config = configparser.ConfigParser()
    for key, dict_ in data.items():
        if not isinstance(dict_, dict):
            raise RuntimeError('辞書のすべての値が辞書型でありません')
        config[key] = dict_
    with open(out_path_ini, 'w', encoding='utf-8') as f:
        config.write(f)
