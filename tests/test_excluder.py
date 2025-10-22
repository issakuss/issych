import unittest
from pathlib import Path

import pandas as pd
import pingouin as pg

from issych.dataclass import Dictm
from issych.excluder import DataExcluder


class TestDataExcluder(unittest.TestCase):
    def setUp(self):
        dataset = pg.read_dataset('penguins')
        dataset.index = [f's{str(i + 1).zfill(3)}'
                         for i in range(len(dataset))]
        cols_to_track = ['body_mass_g', 'sex', 'island']
        self.exc = DataExcluder(dataset, cols_to_track=cols_to_track)

    def test_error(self):
        index_duplicated = pd.DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            columns=['row1', 'row1', 'row2', 'row3']).T
        with self.assertRaises(RuntimeError):
            DataExcluder(index_duplicated)

        with self.assertRaises(RuntimeError):
            range_indexed = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])
            DataExcluder(range_indexed)

        with self.assertRaises(RuntimeError):
            (self.exc.rm_byvalue('same_reason', 'bill_length_mm', '>', 59)
                     .rm_byvalue('same_reason', 'bill_length_mm', '<', 32.2,
                                 tags=['tag1', 'tag2']))

        with self.assertRaises(RuntimeError):
            exc = self.exc.rm_byvalue('reason', 'bill_length_mm', '>', 59)
            exc2 = DataExcluder(pg.read_dataset('penguins')).rm_as(exc, -1, -5)

    def test_success(self):
        exc = (self.exc
               .rm_index('ng_indiv1', ['s000', 's001', 's002'],
                         tags=['tag1'])
               .rm_index('ng_indiv2', 's003')
               .rm_byvalue('high_bill_len', 'bill_length_mm', '>', 59)
               .rm_byvalue('low_bill_len', 'bill_length_mm', '<', 32.2,
                           tags=['tag1', 'tag2'])
               .rm_na('isna_bill_dep', 'bill_depth_mm')
               .rm_byvalue('high_bill_dep', 'bill_depth_mm', '>=', 21.5,
                           tags=['tag1'])
               .rm_byvalue('low_bill_dep', 'bill_depth_mm', '<=', 13.1,
                           tags=['tag2'])
               .rm_byvalue('high_sd_flipper',
                           'flipper_length_mm', '>', '1.5 SD',
                           calc_on='low_bill_dep')
               .rm_byvalue('low_sd_flipper',
                           'flipper_length_mm', '<=', '-1.5 SD',
                           calc_on='low_bill_dep')
               .rm_byvalue('high_iqr_bodymass',
                           'body_mass_g', '>', '1.2 IQR',
                           calc_on=-2)
               .rm_byvalue('low_iqr_bodymass',
                           'body_mass_g', '<=', '-1.2 IQR')
               .rm_index_except('ok_indiv', ['s003', 's004', 's005', 's006'],
                                   tags=['tag1', 'tag2', 'tag3']))

        io_path_temp = Path('excluder_temp.pkl')
        exc.to_pickle(io_path_temp)
        exc = DataExcluder().read_pickle(io_path_temp)
        io_path_temp.unlink()

        summary = exc.get_summary()
        self.assertEqual(
            summary.n_data.tolist(),
            [344, 342, 341, 340, 339, 337, 336, 335, 306, 295, 293, 293, 2])
        self.assertEqual(summary.shape, (13, 15))

        self.assertEqual(len(exc.get_df()), 2)
        self.assertEqual(len(exc.get_df(-2)), 293)
        self.assertEqual(len(exc.get_df(0)), 344)

        self.assertEqual(len(exc.get_ok_index()), 2)
        self.assertEqual(len(exc.get_ok_index(-2)), 293)
        self.assertEqual(len(exc.get_ok_index(0)), 344)

        self.assertEqual(len(exc.get_ng_index()), 291)
        self.assertEqual(len(exc.get_ng_index(-2)), 0)
        self.assertEqual(len(exc.get_ng_index(-3)), 2)
        self.assertEqual(len(exc.get_ng_index(0)), 0)

        self.assertAlmostEqual(
            exc.retention_matrix().loc['ng_indiv1', 'low_bill_dep'],
            0.97953216374269)

        fig = exc.plot_summaries(
            n_col=2, abbr=Dictm(body_mass_g='Body Mass'),
            rcparams_toml='tests/testdata/config/rcparams.toml')
        fig.savefig('tests/output/test_excluder.png')

        dataset2 = pg.read_dataset('penguins')
        dataset2.index = [f'sa{str(i + 1).zfill(3)}'
                         for i in range(len(dataset2))]
        exc2 = DataExcluder(dataset2).rm_as(
            exc, based_from=2, based_to='low_iqr_bodymass')
        self.assertEqual(exc2.get_df().shape, (294, 7))
