from typing import Dict
import configparser
from pathlib import Path

from .typealias import Pathlike
from .dataclass import Dictm


def load_config(ini_path: Pathlike) -> Dictm:
    """
    INIファイルを`issych.dataclass.Dictm`型として読み込みます。
    読み込まれた値は、`eval()`を使って処理されます。
    """

    def except_values(val):
        if val in ['false', 'true']:
            return val.capitalize()
        return val

    def eachsection(parser, section):
        config = Dictm(parser.items(section))
        for key in config:
            config[key] = eval(except_values(config[key]))
        return config

    if not Path(ini_path).exists():
        raise FileNotFoundError(f'Not exist: {ini_path}')

    parser = configparser.ConfigParser()
    parser.read(ini_path)
    return Dictm({section: eachsection(parser, section)
                  for section in parser.sections()})


def save_ini(data: Dict[str, Dict], ini_path: Pathlike):
    """
    INIファイルを保存します。
    """

    config = configparser.ConfigParser()
    for key, dict_ in data.items():
        config[key] = dict_
    with open(ini_path, 'w') as f:
        config.write(f)