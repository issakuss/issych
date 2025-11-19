from typing import Optional, Any, Literal, List
from copy import deepcopy
import operator
import math
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from issych import Dictm, Figure, Pathlike
from issych.stat import nanzscore2value, iqr2value
from issych.figure import prepare_axes, plot_raincloud, get_current_rcparams


INVERSE = {'>': '<=', '>=': '<', '<': '>=', '<=': '>'}
OPERATORS = {'>': operator.gt, '>=': operator.ge,
             '<': operator.lt, '<=': operator.le}


class DataExcluder:
    """
    データセットからデータを除外し、その過程を記録するためのクラスです。
    """

    def __init__(self, df: Optional[pd.DataFrame]=None,
                 cols_to_track: str | Optional[List[str]]=None):
        """
        Parameteres
        -----------
        df : :py:class:`pandas.DataFrame`
            処理するデータセットです。
        cols_to_track : str or list of str
            ここに指定した列名のデータについて、除外の各過程でその統計量を計算し記録します。
            数値の列については平均値と標準偏差を記録します。
            カテゴリカルな列についてはその数を記録します。
        """
        if df is None:
            return # Only for read_pickle()

        self.original_df = df.copy()
        self.df = df.copy()
        self.cols_to_track = cols_to_track if cols_to_track else []

        self._validate()

        self._reasons = []
        self._logged = Dictm(
            ok_indices=Dictm(), ng_indices=Dictm(), summary=[])
        self._log(reason='_init')

    def _copy(self) -> 'DataExcluder':
        new = object.__new__(DataExcluder)

        new.original_df = self.original_df.copy()
        new.df = self.df.copy()
        new.cols_to_track = list(self.cols_to_track)
        new._reasons = list(self._reasons)
        new._logged = deepcopy(self._logged)

        return new

    def _validate(self):
        for col in self.cols_to_track:
            if col not in self.df:
                raise RuntimeError(f'{col}列がデータセットにありません')

        if self.df.index.has_duplicates:
            raise RuntimeError('データセットのindexに重複があります')

        if isinstance(self.df.index, pd.RangeIndex):
            raise RuntimeError('データセットのindexがRangeIndexです')

    def _reason_is_unique(self, reason: str):
        if reason in self._reasons:
            raise RuntimeError('"reason"は重複させないでください。')

    def _log(self, reason: str, col: Optional[str]=None,
             rel: Optional[str]=None, thr: Optional[Any]=None,
             val_thr: Optional[float]=None, ng_index: pd.Index=pd.Index([]),
             tags: Optional[List[str]]=None):
        self._reasons.append(reason)
        self._logged.ok_indices.update({reason: self.df.copy().index})
        self._logged.ng_indices.update({reason: ng_index})

        if len(self._logged.summary) == 0:
            n_attrition = 0
        else:
            n_attrition = self._logged.summary[-1].n_data - len(self.df)
        summary = dict(
            reason=reason, col=col, rel=rel, thr=thr, val_thr=val_thr,
            tags='' if tags is None else '; '.join(tags), n_data=len(self.df),
            n_attrition=n_attrition)

        for col in self.cols_to_track:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                summary[f'mean__{col}'] = self.df[col].mean()
                summary[f'std__{col}'] = self.df[col].std()
            else:
                ns = self.df[col].value_counts().to_dict()
                summary |= {f'n__{k}': v for k, v in ns.items()}

        self._logged.summary.append(pd.Series(summary))

    def _calc_sd_threshold(self, col: str, thr: str,
                           calc_on: str | int=-1) -> float:
        val = float(thr.split(' SD')[0])
        return nanzscore2value(val, self.get_df(calc_on)[col])

    def _calc_iqr_threshold(self, col: str, thr: str,
                            calc_on: str | int=-1) -> float:
        val = float(thr.split(' IQR')[0])
        return iqr2value(val, self.get_df(calc_on)[col])

    def to_pickle(self, ot_path: Pathlike, **kwargs: dict):
        with open(ot_path, 'wb') as f:
            pickle.dump(self, f, **kwargs)

    def read_pickle(self, in_path: Pathlike, **kwargs: dict
                    ) -> 'DataExcluder':
        with open(in_path, 'rb') as f:
            return pickle.load(f, **kwargs)

    def rm_byvalue(self, reason: str,
                   col: str, rel: Literal['>', '>=', '<', '<='],
                   thr: str | float, calc_on: str | int=-1,
                   tags: Optional[List[str]]=None
                   ) -> 'DataExcluder':
        """
        ``col`` 列の値が ``thr`` で指定した値に対して ``rel`` であるデータを除外します。

        Parameters
        ----------
        reason : str
            除外の理由を文字列を指定してください。
            本除外についてのIDのようなものです。
            ほかの除外における ``reason`` と重複しないようにする必要があります。
        col : str
            除外の条件にする列名を指定してください。
        rel : {'>', '>=', '<', '<='}
            除外の条件を指定してください。
            例えば ``>`` を指定すると、 ``col`` 列の値が ``val`` より大きいデータが除外されます。
        thr : str or float
            除外の条件にする値を指定してください。
            数値を指定すると、その数値をもとに除外を行います。
            ``"-2.5 SD"`` など、 ``" SD"`` を含む文字列を指定すると、標準偏差をもとに除外を行います。
            ``"1.5 IRQ"`` など、 ``" IQR"`` を含む文字列を指定すると、四分位範囲をもとに除外を行います。
        calc_on: str
            文字列または整数を指定して、``reason`` を指定してください。
            指定された ``reason`` に対応するデータフレームにおいて SD や IQR を計算します。
        tags : list of str, optional
            本除外について、タグを設定できます。
        """
        self._reason_is_unique(reason)
        ex = self._copy()
        if rel not in INVERSE:
            raise RuntimeError(f'relの設定が不正です：{rel}')

        if isinstance(thr, (float, int)):
            val_thr = thr
        elif ' SD' in thr:
            val_thr = ex._calc_sd_threshold(col, thr, calc_on)
        elif ' IQR' in thr:
            val_thr = ex._calc_iqr_threshold(col, thr, calc_on)
        else:
            raise RuntimeError(f'thrの値が不正です：{thr}')

        if isinstance(calc_on, int) and (calc_on != -1):
            thr = f'{thr} (@{ex._reasons[calc_on]})'
        elif isinstance(calc_on, str):
            thr = f'{thr} (@{calc_on})'

        rel_include = INVERSE[rel]
        na_is_ok = f'| ({col}.isnull())'
        ex.df = ex.df.query(f'({col} {rel_include} @val_thr)' + na_is_ok)

        ng_index = ex.get_df(-1).index.difference(ex.df.index)

        ex._log(reason, col=col, rel=rel, val_thr=val_thr, thr=thr,
                tags=tags, ng_index=ng_index)
        return ex


    def rm_na(self, reason: str, col: str, tags: Optional[List[str]]=None
              ) -> 'DataExcluder':
        """
        ``col`` 列の値が NaN のデータを除外します。

        Parameters
        ----------
        reason : str
            除外の理由を文字列を指定してください。
            本除外についてのIDのようなものです。
            ほかの除外における ``reason`` と重複しないようにする必要があります。
        col : str
            除外の条件にする列名を指定してください。
        tags : list of str, optional
            本除外について、タグを設定できます。
        """
        self._reason_is_unique(reason)
        ex = self._copy()

        ex.df = ex.df.dropna(subset=col)
        ng_index = ex.get_df(-1).index.difference(ex.df.index)
        ex._log(reason, col=col, rel='is', val_thr=pd.NA, thr='NaN',
                tags=tags, ng_index=ng_index)
        return ex

    def rm_index(self, reason: str, ng_index: Any,
                 tags: Optional[List[str]]=None):
        """
        指定した ``index`` のデータを除外します。

        Parameters
        ----------
        reason : str
            除外の理由を文字列を指定してください。
            本除外についてのIDのようなものです。
            ほかの除外における ``reason`` と重複しないようにする必要があります。
        ng_index: list of str, optional
            ここに指定したIDを持つデータ提供者を除外します。
        tags : list of str, optional
            本除外について、タグを設定できます。
        """
        self._reason_is_unique(reason)
        ex = self._copy()
        if isinstance(ng_index, str):
            ng_index = [ng_index]

        ex.df = ex.df[~ex.df.index.isin(ng_index)]
        ex._log(reason, col='index', rel='in', thr='; '.join(ng_index),
                tags=tags, ng_index=ng_index)

        return ex

    def rm_index_except(self, reason: str, ok_index: Any,
                        tags: Optional[List[str]]=None):
        """
        指定した ``index`` 以外のデータを除外します。

        Parameters
        ----------
        reason : str
            除外の理由を文字列を指定してください。
            本除外についてのIDのようなものです。
            ほかの除外における ``reason`` と重複しないようにする必要があります。
        ok_index: list of str, optional
            ここに指定したIDを持たないデータ提供者を除外します
        tags : list of str, optional
            本除外について、タグを設定できます。
        """
        self._reason_is_unique(reason)
        ex = self._copy()
        if isinstance(ok_index, str):
            ok_index = [ok_index]

        ex.df = ex.df[ex.df.index.isin(ok_index)]
        ng_index = ex.get_df(-1).index.difference(ex.df.index)
        ex._log(reason, col='index', rel='not in', thr='; '.join(ok_index),
                tags=tags, ng_index=ng_index)

        return ex

    def get_summary(self) -> pd.DataFrame:
        """
        各除外過程における除外理由、条件、除外後におけるデータの様子 (``cols_to_track`` で指定した列の情報) をデータフレームで返します。
        """
        return pd.DataFrame(self._logged.summary)

    def get_ok_index(self, where: int | str=-1) -> pd.Series:
        """
        指定した除外過程において、生き残ったデータの ``index`` を返します。

        Parameters
        ----------
        where : int or str, default=-1
            除外過程を指定してください。
            何番目の除外かを ``int`` で指定するか、``reason`` によって示してください。
            デフォルトは ``-1`` なので、最新のデータが返ります。
        """
        if isinstance(where, int):
            return self._logged.ok_indices[self._reasons[where]]
        return self._logged.ok_indices[where]

    def get_ng_index(self, where: int | str=-1) -> pd.Series:
        """
        指定した除外過程において、除外したデータの ``index`` を返します。

        指定した過程までの全除外データではなく、その過程における除外データについて返します。

        Parameters
        ----------
        where : int or str, default=-1
            除外過程を指定してください。
            何番目の除外かを ``int`` で指定するか、``reason`` によって示してください。
            デフォルトは ``-1`` なので、最新の除外データが返ります。
        """
        if isinstance(where, int):
            return self._logged.ng_indices[self._reasons[where]]
        return self._logged.ng_indices[where]

    def get_df(self, where: int | str=-1) -> pd.DataFrame:
        """
        指定した除外過程（その除外をし終わった時点）におけるデータフレームを返します。

        Parameters
        ----------
        where : int or str, default=-1
            除外過程を指定してください。
            何番目の除外かを ``int`` で指定するか、``reason`` によって示してください。
            デフォルトは ``-1`` なので、最新のデータフレームが返されます。
        """
        return self.original_df.loc[self.get_ok_index(where)]

    def retention_matrix(self) -> pd.DataFrame:
        """
        各除外過程において、以前の過程から比べてデータ数がどれだけ維持されたかを示す正方行列を返します。
        """
        reasons = self._reasons
        ns = np.asarray(self.get_summary().n_data, dtype=float)

        with np.errstate(divide='ignore', invalid='ignore'):
            mat = ns[None, :] / ns[:, None]
            mat[~np.isfinite(mat)] = np.nan

        return (pd.DataFrame(mat, index=reasons, columns=reasons)
                .astype('Float64'))

    def rm_as(self, based: 'DataExcluder | pd.DataFrame',
              based_from: Optional[int | str]=None,
              based_to: Optional[int | str]=None) -> 'DataExcluder':
        def convert_base(base: Optional[int | str], reasons: List[str]
                         ) -> int | None:
            if isinstance(base, int):
                return base
            elif isinstance(base, str):
                if not base in reasons:
                    raise RuntimeError(f'{base} というreasonは存在しません')
                return reasons.index(base)
            elif base is None:
                return None
            else:
                raise RuntimeError(
                    f'{type(base)} はbased_fromまたはbased_toとして使えません。')

        def validate_base(base: Optional[int]):
            if base is None:
                return base
            if base < 0:
                raise RuntimeError('based_fromとbased_to に負数は使用できません')
            return base

        if isinstance(based, pd.DataFrame):
            summary = based.copy()
        else:
            based_from = validate_base(
                convert_base(based_from, based._reasons))
            based_to = validate_base(convert_base(based_to, based._reasons))
            summary = based.get_summary().loc[based_from:based_to, :]

        ex = self._copy()
        for _, row in summary.iterrows():
            tags = row.tags.split('; ')
            if row.col is None:
                ...
            elif row.col == 'index' and row.rel == 'in':
                ex = ex.rm_index(row.reason, row.thr.split('; '), tags)
            elif row.col == 'index' and row.rel == 'not in':
                ex = ex.rm_index_except(row.reason, row.thr.split('; '), tags)
            elif pd.isna(row.val_thr):
                ex = ex.rm_na(row.reason, row.col, tags)
            else:
                ex = ex.rm_byvalue(
                    row.reason, row.col, row.rel, row.val_thr, tags)

        return ex

    def plot_summaries(self, n_col: int=1, abbr: Optional[Dictm]=None,
                       coef_xsize: float=0.4,
                       rcparams_toml: Optional[Pathlike]=None) -> Figure:
        """
        :py:func:`rm_byvalue` による各除外過程について、プロットします。

        その除外過程において用いた列の分布を RainCloud プロットし、除外に用いた基準線を引きます。

        Parameters
        ----------
        n_col : int, default=1
            プロットの段組み数を指定してください。
            デフォルトの1だと、すべてのプロットが縦に並びます。
        abbr : `Dictm`
            変数名と正式名の対応関係を記述した `Dictm` です。
            これを指定すると、プロット中のラベルが正式名に置き換わります。
        coef_xsize : float, default=0.4
            生成される図の横幅を決める係数です。
        rcparams_toml : `Pathlike` 
            図の見た目を決めるための設定を記入した .toml ファイルへのパスを指定してください。
            詳細は :ref:`issych-figure-setting-format` を確認してください。
        """
        target = self.get_summary().query('val_thr.notnull()') # type: ignore
        n_row = math.ceil(len(target) / n_col)
        fig, axes = prepare_axes(n_row=n_row, n_col=n_col,
                                 axsizeratio=(coef_xsize, 1.),
                                 in_path_toml=rcparams_toml)
        color = get_current_rcparams().color
        for ax, (_, row) in zip(axes.flatten(), target.iterrows()):
            df = self.get_df(self._reasons.index(row.reason) - 1).copy()
            df['are_ng'] = OPERATORS[row.rel](df[row.col], row.val_thr)
            plot_raincloud(df[row.col].dropna(), ax=ax,
                           kwargs_strip={'alpha': 0.0})
            sns.stripplot(df, y=row.col, hue='are_ng', legend=False,
                          palette=[color.main, color.sub], ax=ax)
            ax.hlines(
                y=row.val_thr, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1])
            if abbr is not None:
                ax.set_ylabel(abbr.may(ax.get_ylabel()))

        plt.tight_layout()

        return fig
