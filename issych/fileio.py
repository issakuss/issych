import configparser
from pathlib import Path

from .typealias import Pathlike
from .dataclass import Dictm


def load_config(ini_path: Pathlike) -> Dictm:
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

    if not Path(ini_path).exists():
        raise FileNotFoundError(f'Not exist: {ini_path}')

    parser = configparser.ConfigParser()
    parser.read(ini_path)
    return Dictm({section: eachsection(parser, section)
                  for section in parser.sections()})
