from typing import Any, Union, Optional, Literal, Tuple, List, Dict
from copy import deepcopy

import numpy as np
import pandas as pd

from .typealias import Pathlike
from .dataclass import Dictm
from .dataframe import (
    loc_byalphabet, loc_range_byalphabet, loc_cols_name_startswith)


class CantReverseError(Exception):
    def __init__(self):
        super().__init__(
            'min_plus_maxが指定されていないため、逆転項目の処理ができません')


def score_questionnaire(
    answer: pd.DataFrame,
    questname: str,
    preprocess: str='',
    idx_reverse: Optional[List[int]]=None,
    min_plus_max: Optional[int]=None,
    na_policy: Literal['nan', 'ignore', 'raise']='nan',
    subscale: Optional[Dict[str, Union[List[int], Literal['all']]]]=None,
    average: bool=False) -> pd.DataFrame:
    """
    一つの質問紙回答データを集計します。

    Parameters
    ----------
    answer : :py:class:`pandas.DataFrame`
        ある一つの質問紙への回答データです。
        shape=(質問紙への回答数, 質問紙の項目数) となります。
        ``np.nan`` , ``float('nan')`` , ``None`` , ``''`` （空白の文字列）は、 ``pd.NA`` に変換されます。
        数値に変換できないデータが入っていると、一切処理されずに ``answer`` のまま返されます。
        
        .. warning::
            回答者IDやタイムスタンプなど、質問紙への回答以外の列は含めないでください。
            後述の引数で指定する項目番号などにズレが生じる危険があります。
            そういった列は事前に除いてください。
            または、:meth:`pandas.DataFrame.set_index()` で ``index`` に指定してください。

    questname : str
        質問紙の名前です。
        返り値における列名に用いられます。
        わかりやすい名前を自由につけてください。

    preprocess : str, default ''
        質問紙への回答データについて、事前の処理が必要な場合はそのコードを記述してください。
        記述されたコードが、回答データの各要素において実行されます。
        その際、回答データの各要素は ``q`` という変数で扱われます。
        ``pd.isna(q)`` の場合（要素が無回答の場合）、ここに指定された処理は行われません。
        ``preprocess`` に空白の文字列（ ``''`` ）が指定された場合、事前の処理は行われません。

        **Examples**
        
        >>> preprocess='{1: True, 2: False, 3: False}[q]'

        1という回答はTrue, 2または3という回答はFalseになるよう変換する例です。
        性別（is_male）についての回答などに使えます。

        >>> preprocess='int(q > 2)'

        3より大きい数値が回答されていた場合は1、それ以外は0に変換する例です。
        Autism-Spectrum Quotient の回答などに使えます。

        >>> preprocess='int(q == 2)'

        2という回答の場合は1、それ以外は0に変換する例です。

        >>> preprocess='7 - q'

        1〜6点の6件法による質問への回答を、すべて逆転させる例です。
        逆転項目の処理は後述の ``idx_reverse`` で行います。
        逆転項目が通常項目より多い場合には、まずこの処理ですべての項目を逆転させ、
        通常項目を ``idx_reverse`` に指定すると便利です。

        >>> preprocess='1'

        回答内容に関わらず、すべての項目が1に変換されます。

    idx_reverse : list, default []
        逆転項目を（0ではなく）1始まりの数字で指定してください。
        1始まりなので、論文等で報告される逆転項目番号がそのまま使用できます。

        .. warning::
            ``answer`` に質問紙の項目以外の項目が混じっている場合、ズレが生じます。
            たとえば一列目に回答者ID、二列目以降に質問紙への回答がある場合などです。

    min_plus_max : int, default None
        回答が取りうる最小値と最大値の合計値を指定してください。
        逆転項目として指定された項目の得点を逆転させる際、計算に用います。
        たとえば1〜6点までの6件法の場合は（1+6により）7を指定してください。
        1〜4点までの4件法で回答を求め、回答を0〜1に変換してから集計する場合（AQなど）は、
        ``preprocess`` 引数に必要な変換コードを書いたうえで（0+1により）1を指定してください。

    na_policy : {'na', 'ignore', 'raise'}, default 'na'
        ``answer`` に無回答（ ``np.nan`` or ``float('na')`` or ``None`` or ``pd.NA`` ）があった場合の挙動です。

        - 'na': その回答を含む尺度得点・下位尺度得点が ``pd.NA`` になります
        - 'ignore': その回答を除いて合計点または平均点が算出されます
        - 'raise': エラーが返されます

    subscale : dict, default {}
        下位尺度を辞書で指定してください。
        ``None`` の場合は、質問紙の合計得点（または平均）のみが算出されます。
        辞書のキーにはその下位尺度の名前を指定してください。
        値には、その下位尺度が含む項目番号を（0ではなく）1始まりで指定してください。
        1始まりなので、論文等で報告される項目番号がそのまま使用できます。

        または、値に ``'all'`` を指定することもできます。
        その場合は、すべての項目がその下位尺度の得点算出に使われます。

        項目番号にマイナスをつけると（-1 など）、その番号の得点を逆転させたうえで算出します。
        ``idx_reverse`` に指定した項目にマイナスをつけると、逆転の逆転で順項目となります。
        たとえばある項目について、合計得点では逆転項目として扱い、
        下位尺度得点では順項目として扱う場合などに便利です。

        **Examples**

        >>> {'sample': {subscale = {'syakudo_a': [-1, 2, 3, 4],
                                    'syakudo_b': [5, 6, 7, 8],
                                    'goukei': 'all'}}

        'syakudo_a'、'syakudo_b'、'goukei' という下位尺度得点を算出します。
        算出されるデータフレームは、以下のようになります。

        - 'sample__syakudo_a' 列: ``answer.iloc[[0, 1, 2, 3]]`` の合計点
        - 'sample__syakudo_b' 列: ``answer.iloc[[4, 5, 6, 7]]`` の合計点
        - 'sample__syakudo_total' 列: ``answer`` の合計点。

        'sample__syakudo_total' を算出するときにはすべての得点をそのまま使います。
        'sample__syakudo_a' を算出するときには項目1の得点が逆転されます。

    average: bool, default False
        ``True`` が指定されると、合計得点の代わりに平均点が算出されます。

    Notes
    -----
    はじめに ``preprocess`` 引数に記述された処理がなされ、
    次に ``idx_reverse`` 引数で指定された項目が逆転項目として処理されます。
    続いて、そのほかの引数に指定された内容に従って、合計または平均得点が算出されます。
    """
    def reverse(answer: pd.DataFrame, idx_reverse: List[int], min_plus_max: int
                ) -> pd.DataFrame:
        if min_plus_max is None:
            raise CantReverseError

        idx_reverse_ = np.array(idx_reverse, dtype=int) - 1
        reversed_answer = min_plus_max - answer.iloc[:, idx_reverse_]
        answer[answer.columns[idx_reverse_]] = reversed_answer
        return answer.astype('Float64')

    def total(answer: pd.DataFrame, na_policy: str, average: bool, name: str):
        if na_policy == 'raise' and answer.isna().any().any():
            raise ValueError('データにNAが含まれています。\n'
                             'na_policyの設定によって無視することもできます。')
        skipna = na_policy == 'ignore'

        score = answer.sum(axis=1, skipna=skipna)
        if average:
            score = answer.mean(axis=1, skipna=skipna)
        score.name = name
        return score

    def total_subscale(answer: pd.DataFrame, questname: str,
                       min_plus_max: Optional[int], na_policy: str,
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
                raise CantReverseError
            reversed_answer = min_plus_max - answer.iloc[:, idx_rev]
            answer_[answer.columns[idx_rev]] = reversed_answer
            return answer_, abs(idx_)

        def _eachsubscale(answer: pd.DataFrame, questname: str,
                          min_plus_max: int, na_policy: str, average: bool,
                          subscalename: str,
                          idx_include: Union[List[int], Literal['all']]
                          ) -> pd.Series:
            if idx_include == 'all':
                idx_include = list(range(answer.shape[1]))
            answer_, idx_include_ = _additional_reverse(
                answer, idx_include, min_plus_max)
            name = questname + '__' + subscalename
            try:
                return total(answer_.iloc[:, idx_include_ - 1], na_policy,
                             average, name)
            except Exception as exc:
                raise RuntimeError(f'{name} の設定が不正です。') from exc

        return pd.concat([_eachsubscale(answer, questname, min_plus_max,
                                        na_policy, average, *subscale_)
                          for subscale_ in subscale.items()], axis=1)

    if idx_reverse is None:
        idx_reverse = []

    try:
        answer = answer.copy().astype('Float64')
        answer[np.isnan(answer)] = pd.NA
    except ValueError:
        return answer

    if preprocess != '':
        are_na = pd.isna(answer)
        answer = (answer.map(lambda q: eval(preprocess), na_action='ignore')
                  .astype('Float64'))
        answer[are_na] = pd.NA  # .map() sometimes convert pd.NA to np.nan
    if len(idx_reverse) > 0:
        if min_plus_max is None:
            raise CantReverseError
        answer = reverse(answer, idx_reverse, min_plus_max)
    if subscale is None:
        return total(answer, na_policy, average, questname)
    return total_subscale(
        answer, questname, min_plus_max, na_policy, subscale, average)


class Monshi:
    def __init__(self, answer_sheet: pd.DataFrame):
        """
        Googleフォームなどで収集した、心理学的な質問紙への回答を集計するためのクラスです。

        Parameters
        ----------
        answer_sheet : :py:class:`pandas.DataFrame`
            集計対象となる全回答者の全回答を含んだデータフレームです。
            合計や平均などの集計対象となる列は、すべて数字のみから成る必要があります。
            文字列などから成る列は、集計されずにそのまま返されます。
            
            .. hint:: Q. 数字にしたはずなのに集計されない

                A. 数字が文字型として列に含まれている可能性があります。
                該当する列をfloat型に変換してから試してください。

                >>> answer_sheet.astype({'syukei_sarenai_retsu': float})
        """
        self.answer_sheet = answer_sheet
        self._labels = []

    def __getattr__(self, name: str) -> Any:
        return self.get(name)

    def separate(self, cols_item: Dict[str, Tuple[str, str]]):
        """
        回答を集めたデータフレームを、 ``cols_item`` に従って分割します。

        Parameters
        ----------
        cols_item : dict
            辞書のキーには質問紙の名前を指定してください。
            ここで指定した質問紙名が、集計されたデータフレームで用いられます。

            .. hint:: 集計後のデータに回答者IDやタイムスタンプなどを含む列を残したい場合

                'info'や'meta'など、それとわかる名前を指定しましょう。

            値には、その質問紙に該当する列を指定します。
            ４通りの指定方法があります。

            - 1. 列を1始まりの数字で指定する

            >>> [3, 4, 5, 6, 7]

            数字が1から始まることに注意してください。
            一番左の列は0ではなく1です。
            :meth:`pandas.DataFrame.iloc` などで指定する際と異なります。

            - 2. 列をアルファベットで指定する

            >>> ['H', 'J']

            とすると、7列目と9列目が指定されます

            - 3. 列範囲をアルファベットで指定する

            >>> 'H:J'

            とすると、7, 8, 9列目が指定されます。

            - 4. 列名の冒頭文字列で指定する

            >>> 'scale2_'

            とすると、データフレーム列名が 'scale2\_' から始まる列が指定されます。

        Notes
        -----
        ``.separate()`` を実行すると、質問紙ごとの回答を取得できるようになります。
        指定した質問紙名がそのための attribute となります。

        Examples
        --------
        >>> cols_item = dict(
        >>>     info=[0, 1]
        >>>     scale1=['C', 'D', 'E', 'G', 'H']
        >>>     scale2='scale2_')
        >>>
        >>> monshi = Monshi(answer_sheet)
        >>> monshi.separate(cols_item)
        >>> monshi.shape
        (5, 21)
        >>> monshi.info.shape
        (5, 2)
        >>> monshi.scale1.shape
        (5, 5)
        """
        cols_item = deepcopy(dict(cols_item))
        self._labels = cols_item.keys()

        for key, value in cols_item.items():
            val_is_all_int = all(isinstance(x, int) for x in value)
            val_is_all_str = all(isinstance(x, str) for x in value)
            match value:
                case list() | tuple() if val_is_all_int:
                # Example: [1, 2, 3, 4, 5]
                    value_ = [v - 1 for v in value]
                    separated = self.answer_sheet.iloc[:, value_]
                case list() | tuple() if val_is_all_str:
                # Example: ['H', 'J']
                    separated = loc_byalphabet(self.answer_sheet, value)
                case str() if ':' in value:
                # Example 'H:J'
                    separated = loc_range_byalphabet(
                        self.answer_sheet, *value.split(':'))
                case str():
                # Example 'scale2_'
                    separated = loc_cols_name_startswith(
                        self.answer_sheet, value)
                case _:
                    raise RuntimeError('範囲の指定方法が不正です')

            setattr(self, key, separated)
        return self

    def score(self, monfig: Dict[str, Dict]):
        """
        :py:meth:`Monshi.separate` メソッドで分割された質問紙ごとに集計を行います。
        
        Parameters
        ----------
        monfig: dict
            この引数で、集計方法を質問紙ごとに指定してください。
            ``monfig`` のキーには :py:meth:`Monshi.separate` で指定した質問紙名を記入してください。
            指定されていない質問紙名のキーに対応する値は無視されます。
            ``monfig`` の値は、キーワード引数として :py:func:`score_questionnaire` に与えられます。
            詳細は :py:func:`score_questionnaire` を確認してください。

            .. tip:: Monshiのconfigなので、monfigです
        """

        if len(self._labels) == 0:
            raise RuntimeError('.separate()が先に実行されていません。')
        scores = []
        for label in self._labels:
            if label not in monfig:
                raise RuntimeError(f'{label}についての設定がmonfigにありません。')
            params = monfig[label]
            scores.append(score_questionnaire(
                getattr(self, label), label, **params))
        self._scores = pd.concat(scores, axis=1)
        return self._scores


def score_as_monfig(answer_sheet: pd.DataFrame,
                    monfig: Optional[Dict[str, Dict]] | Optional[Pathlike]
                    ) -> pd.DataFrame:
    """
    Googleフォームなどで収集した、心理学的な質問紙への回答を集計する関数です。
    :class:`Monshi` を使うより手っ取り早いです。

    .. warning::
        このコードが正しく動作することを、作者は保証しません。
        このコードを用いて集計する際は、必ず別の方法による集計結果と比較してください。

    Parameters
    ----------
    answer_sheet : :py:class:`pandas.DataFrame`
        集計対象となる全回答者の全回答を含んだデータフレームです。 
        詳細は、 :meth:`Monshi.__init__` のドキュメントを参照してください。
    monfig : dict or :py:data:`issych.typealias.Pathlike`
        集計方法を示した、 ``.toml`` ファイルへのパスまたは辞書です。
        ``monfig`` には、 ``_cols_item`` というキーが含まれている必要があります。
        ``_cols_item`` は、質問紙名とその質問紙への回答を含む列の指定から成る辞書です。
        詳細は :meth:`Monshi.separate` ドキュメントの、 ``cols_item`` を参照してください。
        また ``monfig`` は、 ``_cols_item`` で指定したすべての質問紙名をキーとして含む必要があります。
        詳細は :func:`score_questionnaire` のドキュメントを参照してください。

    Returns
    -------
    score : :py:class:`pandas.DataFrame`
        集計された得点のデータフレームです。
        詳細は、 :meth:`Monshi.score` のドキュメントを参照してください。

    Notes
    -----
    この関数は...

    >>> Monshi(answer_sheet).separate(monfig['_cols_item']).score(monfig)

    ...のショートカットです。
    詳細はそれぞれのメソッドにおけるドキュメントを参照してください。

    See also
    --------
    Monshi: このショートカット関数が使用しているクラス
    """
    if isinstance(monfig, Pathlike):
        monfig = Dictm(monfig)
    return Monshi(answer_sheet).separate(monfig['_cols_item']).score(monfig)
