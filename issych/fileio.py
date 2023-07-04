from typing import Union
import configparser
from pathlib import Path

from .dataclass import Dictm


def load_config(ini_path: Union[Path, str]) -> Dictm:
    """
    Load INI file.
    Loaded values are converted using eval()
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

    parser = configparser.ConfigParser()
    parser.read(ini_path)
    return Dictm({section: eachsection(parser, section)
                  for section in parser.sections()})
