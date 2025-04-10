import unittest

from issych.fileio import save_ini


class TestSaveINI(unittest.TestCase):
    def test(self):
        config = {
            'section1': {
                'key1': 'value1',
                'key2': 'value2'
            },
            'section2': {
                'key3': 'value3'
            }
        }
        save_ini(config, 'tests/output/test.ini')

        with open('tests/output/test.ini', 'r', encoding='utf-8') as f:
            f.read()
