from typing import Optional, Any, Dict, Tuple
import sys
import os
import contextlib

import pandas as pd
from rpy2.robjects import r, pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from issych.typealias import Number
from issych.dataclass import Dictm

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


def rver() -> Dictm:
    """
    使用している R のバージョンを取得します。

    Examples
    --------
    >>> r_version = rver()
    >>> r_version.major
    '4'
    >>> r_version.minor
    '5.1'
    >>> r_version.full
    'R version 4.5.1 (2026-06-13)'
    """
    return Dictm(
        major=r('R.version$major')[0],
        minor=r('R.version$minor')[0],
        full=r('version$version.string')[0])


class GlmmTMB:
    def __init__(self, data: pd.DataFrame, formula: str,
                 family: str='gaussian', verbose: bool=False):
        """
        R パッケージである ``glmm_tmb`` を Python から使用するためのクラスです。
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

        self.data = self._prepdata(data.copy())
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
        self._fitted_ems = False
        self._coefs = pd.DataFrame([])

    def _prepdata(self, data: pd.DataFrame) -> pd.DataFrame:
        object_col = data.select_dtypes('object').columns
        data[object_col] = data[object_col].astype(str)
        return data

    def _raise_no_fitting(self):
        if not self._fitted:
            raise RuntimeError('先に .fit() を実行してください。')

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
                fitness_mat <- as.data.frame(summary(model)$AICtab)
            ''')
        self._coefs = (pandas2ri.rpy2py(r['coef_mat'])
                       .set_axis(['est', 'std', 'z', 'p'], axis=1))
        fitness_colnames = ['aic', 'bic', 'loglikeli', 'loglikeli2', 'resid']
        self._fitness = (pandas2ri.rpy2py(r['fitness_mat'])
                         .set_axis(fitness_colnames).iloc[:, 0].rename(''))

        self._fitted = True
        return self

    def diagnose(self):
        """
        フィッティングの診断を表示します。
        """
        self._raise_no_fitting()
        print(r('diagnose(model)'))

    def get_summary(self):
        """
        フィッティングの概要をテキストで取得します
        """
        self._raise_no_fitting()
        return str(r('summary(model)'))

    def get_fitness(self) -> pd.Series:
        """
        フィッティングの適合度指標を表示します。

        Returns
        -------
        fitness : :py:class:`pandas.DataFrame`
            フィッティングの適合度指標です。
        """
        self._raise_no_fitting()
        return self._fitness

    def get_coefs(self) -> pd.DataFrame:
        """
        フィッティングの係数を表示します。

        Returns
        -------
        coefs : :py:class:`pandas.DataFrame`
            フィッティングの係数です。
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

    def emmeans(self, specs: str, **kwargs) -> pd.DataFrame:
        """
        フィットしたモデルを用いて、R パッケージ ``emmeans`` の ``emmeans`` を使用します。
        詳細は ``emmeans`` のドキュメントを参照してください。
        """
        self._raise_no_fitting()
        rkwargs = kwargs4r(kwargs)
        with _suppress_r_console_output(self._verbose):
            r('library(emmeans)')
            r(f'''
                ems <- emmeans(model, ~ {specs}, {rkwargs})
                ems_df <- as.data.frame(test(ems))
                effsize <-as.data.frame(
                    eff_size(ems, sigma=sigma(model), edf=df.residual(model)))
                ems_df$effsize <- effsize$effect.size
                low <- intersect(names(effsize), c("asymp.LCL", "lower.CL"))[1]
                up <- intersect(names(effsize), c("asymp.UCL", "upper.CL"))[1]
                ems_df$lower <- effsize[[low]]
                ems_df$upper <- effsize[[up]]
                ems_df[] <- lapply(ems_df, function(x) 
                    if(is.factor(x)) as.character(x) else x)
            ''')
        self._fitted_ems = True
        columns = {'SE': 'std', 'z.ratio': 'z', 'p.value': 'p',
                   'estimate': 'est'}
        return pandas2ri.rpy2py(r['ems_df']).rename(columns, axis=1)

    def emtrends(self, specs: str, var: str,
                 at: Optional[Dict[str, Dict[str, Number]]]=None, **kwargs
                 ) -> pd.DataFrame:
        """
        フィットしたモデルを用いて、R パッケージ ``emmeans`` の ``emtrends`` を使用します。
        詳細は ``emmeans`` のドキュメントを参照してください。
        """

        self._raise_no_fitting()
        rkwargs = kwargs4r(kwargs)
        if at is not None:
            rkwargs += f', at = list({kwargs4r(at)})'
        with _suppress_r_console_output(self._verbose):
            r('library(emmeans)')
            r(f'''
                ems <- emtrends(model, ~ {specs}, var="{var}", {rkwargs})
                ems_df <- as.data.frame(test(ems))
                effsize <-as.data.frame(
                    eff_size(ems, sigma=sigma(model), edf=df.residual(model)))
                ems_df$effsize <- effsize$effect.size
                low <- intersect(names(effsize), c("asymp.LCL", "lower.CL"))[1]
                up <- intersect(names(effsize), c("asymp.UCL", "upper.CL"))[1]
                ems_df$lower <- effsize[[low]]
                ems_df$upper <- effsize[[up]]
                ems_df[] <- lapply(ems_df, function(x) 
                    if(is.factor(x)) as.character(x) else x)
            ''')

        self._fitted_ems = True
        columns = {'SE': 'std', 't.ratio': 't', 'z.ratio': 'z', 'p.value': 'p'}
        return pandas2ri.rpy2py(r['ems_df']).rename(columns, axis=1)

    def contrast(self, **kwargs) -> pd.DataFrame:
        """
        フィットしたモデルを用いて、R パッケージ ``emmeans`` の ``contrast`` を使用します。
        詳細は ``emmeans`` のドキュメントを参照してください。
        """
        if not self._fitted_ems:
            raise RuntimeError(
                '先に .emtrends() か .emmeans() を実行してください。')
        with _suppress_r_console_output(self._verbose):
            r(f'''
                contrast <- contrast(ems, {kwargs4r(kwargs)})
                result <- as.data.frame(contrast)
                effsize <-as.data.frame(eff_size(
                    contrast, sigma=sigma(model), edf=df.residual(model)))
                result$effsize <- effsize$effect.size
                low <- intersect(names(effsize), c("asymp.LCL", "lower.CL"))[1]
                up <- intersect(names(effsize), c("asymp.UCL", "upper.CL"))[1]
                result$lower <- effsize[[low]]
                result$upper <- effsize[[up]]
                result[] <- lapply(result, function(x) 
                    if(is.factor(x)) as.character(x) else x)
            ''')
        cols = {'estimate': 'est', 'SE': 'std', 't.ratio': 't', 'p.value': 'p'}
        return (pandas2ri.rpy2py(r['result']).rename(cols, axis=1))
