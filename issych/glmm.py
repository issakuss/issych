from typing import Optional, Tuple, Dict
import sys
import os
import contextlib

import pandas as pd
from rpy2.robjects import r, pandas2ri, default_converter, FloatVector
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from issych.typealias import Number

from .stat import kwargs4r


@contextlib.contextmanager
def _suppress_r_console_output(verbose: bool):
    if verbose:
        yield
        return
    with open(os.devnull, 'w') as devnull:  # type: ignore
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class GlmmTMB:
    def __init__(self, data: pd.DataFrame, formula: str,
                 family: str='gaussian', verbose: bool=False):
        """
        R パッケージである ``glmm_tmb`` を Python から使用するためのクラスです。
        :py:class:`rpy2` を用います。
        R に ``glmm_tmb`` がインストールされている必要があります。

        Parameters
        ----------
        data : :py:class:`pandas.DataFrame`
            解析の対象となるデータフレームです。
        formula : str
            ``glmm_tmb`` に渡す数式です。
        family : str
            ``glmm_tmb`` に渡す分布ファミリーです。
        verbose : bool, default False
            ``False`` のとき、R からの出力が抑制されます。

        Examples
        --------
        >>> formula = 'score ~ group * time + (1 | sub_id)'
        >>> result = GlmmTMB(data, formula).fit().summary()
        """
        with _suppress_r_console_output(verbose):
            base = importr("base")  # type: ignore
            utils = importr("utils")  # type: ignore
            glmmTMB = importr("glmmTMB")  # type: ignore
            r('library(glmmTMB)')

        data = self._prepdata(data.copy())
        with localconverter(default_converter + pandas2ri.converter):
            rdata = pandas2ri.py2rpy(data)
        r.assign('data', rdata)
        r('''
            data[] <- lapply(data, function(col) {
            if (is.character(col)) as.factor(col) else col
            })
          ''')

        self.formula = formula
        self.family = family

        self._verbose = verbose
        self._fitted = False
        self._fitted_emtrend = False
        self._coefs = pd.DataFrame([])

    def _prepdata(self, data: pd.DataFrame) -> pd.DataFrame:
        object_col = data.select_dtypes('object').columns
        data[object_col] = data[object_col].astype(str)
        return data

    def fit(self):
        """
        フィッティングを実施します。
        """
        with _suppress_r_console_output(self._verbose):
            r(f'''
                model <- glmmTMB(
                    formula = {self.formula},
                    data = data,
                    family = {self.family}()
                )
                coef_mat <- as.data.frame(summary(model)$coefficients$cond)
                coef_rownames <- rownames(coef_mat)
            ''')
        self._coefs = (pandas2ri.rpy2py(r['coef_mat'])
                       .set_axis(['est', 'std', 'z', 'p'], axis=1))
        self._fitted = True
        return self

    def summary(self) -> pd.DataFrame:
        """
        フィッティングの結果を表示します。

        Returns
        -------
        fitted : :py:class:`pandas.DataFrame`
            フィッティングの結果です。
        """
        if not self._fitted:
            raise RuntimeError('先に .fit() を実行してください。')
        return self._coefs

    def sigma(self) -> float:
        """
        モデルの残差標準偏差を返します。
        各項の係数をこれで割ることで、Cohen's d（に近い値）を計算できます。
        """
        if not self._fitted:
            raise RuntimeError('先に .fit() を実行してください。')
        return r('sigma(model)')[0]

    def emtrends(self, specs: str,
                 at: Optional[Dict[str, Dict[str, Number]]]=None, **kwargs
                 ) -> pd.DataFrame:
        """
        フィットしたモデルを用いて、R パッケージ ``emmeans`` の ``emtrends`` を使用します。
        詳細は ``emmeans`` のドキュメントを参照してください。
        """

        rkwargs = kwargs4r(kwargs)
        if at is not None:
            rkwargs += f', at = list({kwargs4r(at)})'
        with _suppress_r_console_output(self._verbose):
            r('library(emmeans)')
            r(f'''
                trends <- emtrends(model, ~ {specs}, {rkwargs})
                trends_df <- as.data.frame(trends)
            ''')

        self._fitted_emtrend = True
        columns = {'SE': 'std', 'lower.CL': 'lower', 'upper.CL': 'upper'}
        return pandas2ri.rpy2py(r['trends_df']).rename(columns, axis=1)

    def contrast(self, **kwargs) -> pd.DataFrame:
        """
        フィットしたモデルを用いて、R パッケージ ``emmeans`` の ``contrast`` を使用します。
        詳細は ``emmeans`` のドキュメントを参照してください。
        """
        if not self._fitted_emtrend:
            raise RuntimeError('先に .emtrends() を実行してください。')
        with _suppress_r_console_output(self._verbose):
            r(f'''
                result <- as.data.frame(contrast(trends, {kwargs4r(kwargs)}))
            ''')
        cols = {'estimate': 'est', 'SE': 'std', 't.ratio': 't', 'p.value': 'p'}
        return (pandas2ri.rpy2py(r['result']).rename(cols, axis=1))
