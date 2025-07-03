from typing import Tuple
import sys
import os
import contextlib

import pandas as pd
from rpy2.robjects import r, pandas2ri, default_converter, FloatVector
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


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

        data = self._prepdata(data)
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
        self._coefs = pd.DataFrame(r['coef_mat'], columns=r['coef_rownames'],
                                   index=['coef', 'std', 'z', 'p']).T
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
        """
        return r('sigma(model)')[0]

    def contrast(self, compareby: str, cutoffs: Tuple[float, float]
                 ) -> pd.DataFrame:
        """
        R パッケージである ``emmeans`` を用いて、2点における係数の差を検定します。
        ``emmeans`` が R にインストールされている必要があります。

        Parameters
        ----------
        compareby : str
            ここに指定した列名について比較を行います。
        cutoff : tuple of float
            ここに指定した2点において比較を行います。

        Returns
        -------
        trends : :py:class:`pandas.DataFrame`
            検定結果です。
        """
        with _suppress_r_console_output(self._verbose):
            r('library(emmeans)')
            dv = self.formula.split(' ')[0]
            r.assign('cutoffs', FloatVector(cutoffs))
            r(f'''
                trends <- emtrends(model, ~ {compareby}, var = {dv},
                                   at = list({compareby} = cutoffs))
                result <- as.data.frame(contrast(trends, method = "revpairwise"))
            ''')

        trends = (pd.DataFrame(r['result'])
                  .set_axis(['coef', 'std', 'df', 'stat', 'p']).T)
        return trends
