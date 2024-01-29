from typing import Union, Optional, Literal, Tuple, List, Dict
from copy import deepcopy

import numpy as np
import pandas as pd

from .fileio import load_config
from .dataframe import extract_byalphabet, extract_colname_startswith


def score_questionnaire(
    answer: pd.DataFrame, questname: str, preprocess: Optional[str]=None,
    idx_reverse: Optional[List[int]]=None, min_plus_max: Optional[int]=None,
    nanpolicy: Literal['nan', 'ignore', 'raise']='nan',
    subscale: Optional[Dict[str, Union[List[int], Literal['all']]]]=None,
    average: bool=False) -> pd.DataFrame:
    """
    answer: pd.DataFrame
        Answer sheet of questionnaires.
    questname: str
        Name of the questionnaire
    preprocess: str | None
        Python code to be applied to each cell of the answer sheet.
    idx_reverse: List[int] | None
        Index of reverse items.
        The index starts with 1, not 0.
    min_plus_max: int | None
        Minimum choice plus maximum choice.
        For example, if the choice of the questionnaire ranges from 1 to 5,
        it is 6.
    nanpolicy: Literal['nan', 'ignore', 'raise']
        Behavior when the data includes empty answers.
        'nan': Calculated score will be NaN.
        'ignore': It ignores NaN and calculates the score.
        'raise': It raises an error.
    subscale: dict | None
        Indicates the subscale.
        Its key is the subscale name.
        Its value is the index of items included in that subscale.
        e.g., 'sub_a': [1, 3, 5]
        Alternatively, you can use 'all' to include all items.
        e.g., 'total': 'all'
        If index is negative value, it reverses the score.
        e.g., 'sub_b': [-1, 4, 6]
        In this case, item 1 will be reverse scored.
        If item 1 is indicated as a reversed one in idx_reverse,
        it will be straightly scored in the calculation of sub_b.
    average: bool
        If True, it calculate the average instead of sum.
    """

    def _preprocess(answer: pd.DataFrame, preprocess: str) -> pd.DataFrame:
        answer = answer.copy()
        col_pdfloat = answer.select_dtypes('Float64').columns
        answer[col_pdfloat] = answer[col_pdfloat].astype(float)  # na_action option will not ignore pd.NA
        try:
            return answer.map(lambda q: eval(preprocess), na_action='ignore')  # type: ignore
        except:
            raise ValueError('Following preprocess code failed: '+ preprocess)

    def _reverse(answer: pd.DataFrame, idx_reverse: List[int],
                 min_plus_max: int) -> pd.DataFrame:
        answer = answer.copy()
        if min_plus_max is None:
            raise ValueError(
                'min_plus_max must be entered when idx_reverse is entered')

        idx_reverse_ = np.array(idx_reverse, dtype=int) - 1
        reversed_answer = min_plus_max - answer.iloc[:, idx_reverse_]
        answer[answer.columns[idx_reverse_]] = reversed_answer
        return answer

    def _total(answer: pd.DataFrame, nanpolicy: str, average: bool, name: str):
        answer = answer.copy().astype('Float64')
        if nanpolicy == 'raise' and answer.isna().any().any():
            raise ValueError('Dataframe includes NaN\n'
                             'You can ignore this by nanpolicy setting')
        skipna = nanpolicy == 'ignore'

        score = answer.sum(axis=1, skipna=skipna)
        if average: 
            score = answer.mean(axis=1, skipna=skipna)
        score.name = name
        return score

    def _total_subscale(answer: pd.DataFrame, questname: str,
                        min_plus_max: Optional[int], nanpolicy: str,
                        subscale: Dict, average: bool) -> pd.DataFrame:

        def _additional_reverse(
            answer: pd.DataFrame, idx_include: List[int],
            min_plus_max: Optional[int]) -> Tuple[pd.DataFrame, np.ndarray]:

            answer_ = answer.copy()
            idx_ = np.array(idx_include, dtype=int)
            idx_rev = abs(idx_[idx_ < 0]) - 1
            if len(idx_rev) == 0:
                return answer, idx_
            if min_plus_max is None:
                raise ValueError('min_plus_max must be entered')
            reversed_answer = min_plus_max - answer.iloc[:, idx_rev]
            answer_[answer.columns[idx_rev]] = reversed_answer
            return answer_, abs(idx_)

        def _eachsubscale(answer: pd.DataFrame, questname: str,
                          min_plus_max: int, average: bool, subscalename: str,
                          idx_include: Union[List[int], Literal['all']]
                          ) -> pd.Series:
            if idx_include == 'all':
                idx_include = list(range(answer.shape[1]))
            answer_, idx_include_ = _additional_reverse(
                answer, idx_include, min_plus_max)
            name = questname + '_' + subscalename
            try:
                return _total(
                    answer_.iloc[:, idx_include_ - 1], nanpolicy, average, name)
            except:
                raise ValueError(f'Found error in {name}. '
                                 'Please check your settings.')

        return pd.concat([
            _eachsubscale(answer, questname, min_plus_max, average, *subscale_)
            for subscale_ in subscale.items()], axis=1)

    if preprocess is not None:
        answer = _preprocess(answer, preprocess)
    if (idx_reverse is not None) and (min_plus_max is not None):
        answer = _reverse(answer, idx_reverse, min_plus_max)
    if subscale is None:
        return _total(answer, nanpolicy, average, questname)
    return _total_subscale(
        answer, questname, min_plus_max, nanpolicy, subscale, average)


class Monshi:
    def __init__(self, answer_sheet: pd.DataFrame):
        self.answer_sheet = answer_sheet

    def _raise_column_duplicated(self, df: pd.DataFrame):
        duplicate_columns = set(df.columns[df.columns.duplicated()])
        if duplicate_columns:
            raise ValueError('Following columns are duplicated: '
                            f'{duplicate_columns} ')

    def label(self, config_separate: Dict[str, Tuple[str, str]]):
        config_separate = deepcopy(dict(config_separate))
        if '_meta' in config_separate:
            meta = extract_byalphabet(
                self.answer_sheet, *config_separate.pop('_meta'))
        else:
            meta = pd.DataFrame([])

        self._labels = config_separate.keys()

        for key, value in config_separate.items():
            if isinstance(value, tuple):
                separated = extract_byalphabet(self.answer_sheet, *value)
            elif isinstance(value, str):
                separated = extract_colname_startswith(
                    self.answer_sheet, value)
            try:
                separated = separated.astype('Float64')
            except ValueError:
                raise ValueError('Values in indicated range must be numeric.')
            separated = pd.concat([meta, separated], axis=1)
            if len(meta) > 0:
                self._raise_column_duplicated(separated)
                separated = separated.set_index(
                    list(separated.columns[:meta.shape[1]]))
            setattr(self, key, separated)
        return self

    def score(self, config_quest: Dict[str, Dict]):
        """
        preprocess -> reverse -> score
        Pass questionnaires those Monshi contains to score_questionnaire.
        config_quest:
            Dictionary including dictionary.
            The items of inner dictionary are passed to score_questionnaire().
        """

        scores = []
        for label in self._labels:
            params = config_quest[label]
            scores.append(score_questionnaire(
                getattr(self, label), label, **params))
        self._scores = pd.concat(scores, axis=1)
        return self

    def get_scores(self) -> pd.DataFrame:
        return self._scores


if __name__ == '__main__':
    config = load_config('testdata/config/monshi.ini')
    # config.range.pop('_meta')
    # config.range._meta = ('A', 'A')
    answer_sheet = pd.read_csv('testdata/monshi/testdata.csv')
    answer_sheet = answer_sheet.map(
        lambda x: x if pd.isna(x) else x.split('.')[0])
    monshi = Monshi(answer_sheet).label(config.range).score(config)
    scores = monshi.get_scores()
    monshi = Monshi(answer_sheet).label(config.range).score(config)  # Check not destructing vars
    scores = monshi.get_scores()
    manual_scores = pd.read_csv('testdata/monshi/testdata-manually-scored.csv',
                                index_col=['sub', 'timestamp'])
    assert scores.equals(manual_scores.astype('Float64'))
    print('Monshi() is OK')