from typing import Optional, Any, Literal, Tuple, List

from cycler import cycler
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle, Ellipse
import seaborn as sns
from dynaconf import Dynaconf

from .dataclass import Dictm
from .typealias import Pathlike, Figure, Axes, Vector
from .stat import Pvalue2SigMark

# Settings

def set_rcparams(in_path_toml: Pathlike='config/rcparams.toml'):
    """
    ``in_path_toml`` に指定されたファイルにある設定を反映させます。

    Parameters
    ----------
    in_path_toml: Pathlike
        ここに指定されたパスにあるファイルを :py:func:`Dynaconf.dynaconf` で読み取ります。
        読み取った設定を、 :py:data:`matplotlib.rcParams` に反映させます。

    Notes
    -----
    ``toml`` ファイルには、 :ref:`issych-figure-setting-format` で記入されている必要があります。
    """
    params = Dynaconf(settings_files=in_path_toml)
    cur = plt.rcParams

    plt.rcParams.update(
        {'axes.spines.right': False,
         'axes.spines.top': False,
         'errorbar.capsize': params.size.capsize or cur['errorbar.capsize'],
         'figure.dpi': params.misc.dpi or cur['figure.dpi'],
         'figure.figsize': params.size.figsize or cur['figure.figsize'],
         'font.size': params.size.fontsize or cur['font.size'],
         'savefig.format': params.misc.format or cur['savefig.format'],
         'axes.edgecolor': params.color.main or cur['axes.edgecolor'],
         'axes.facecolor': params.color.back or cur['axes.facecolor'],
         'axes.labelcolor': params.color.main or cur['axes.labelcolor'],
         'axes.prop_cycle': cycler(color=params.color.cycle) or \
                            cur['axes.prop_cycle'],
         'figure.edgecolor': params.color.main or cur['figure.edgecolor'],
         'figure.facecolor': params.color.back or cur['figure.facecolor'],
         'ytick.color': params.color.main or cur['ytick.color'],
         'xtick.color': params.color.main or cur['xtick.color'],
         'savefig.facecolor': params.color.back or cur['savefig.facecolor'],
         'lines.color': params.color.main or cur['lines.color'],
         'lines.linewidth': params.size.linewidth or cur['lines.linewidth'],
         'text.color': params.color.main or cur['text.color'],
         'patch.facecolor': params.color.highlight or cur['patch.facecolor'],
         'patch.edgecolor': params.color.subhighlight or \
                            cur['patch.edgecolor'],
         'grid.color': params.color.sub or cur['grid.color'],
         })

    for k, v in params.kwargs.items():
        plt.rcParams[k] = v


def get_current_rcparams() -> Dictm:
    """
    現在の ``rcparams`` を、 :ref:`issych-figure-setting-format` で取得します。

    Returns
    -------
    current_rcparams: Dictm
        以下の三つのキー・値を持つ、ネストされた :class:`issych.dataclass.Dictm` です。

        - ``color``: 色に関する情報です。  
        - ``size``: 図やオブジェクトのサイズに関する情報です。  
        - ``misc``: 図の解像度と、図のファイル拡張子についてです。
    """
    color = Dictm(
        main=plt.rcParams['axes.edgecolor'],
        back=plt.rcParams['figure.facecolor'],
        sub=plt.rcParams['grid.color'],
        highlight=plt.rcParams['patch.facecolor'],
        subhighlight=plt.rcParams['patch.edgecolor'],
        cycle=list(plt.rcParams['axes.prop_cycle']))

    size = Dictm(
        figsize=plt.rcParams['figure.figsize'],
        fontsize=plt.rcParams['font.size'],
        linewidth=plt.rcParams['lines.linewidth'],
        capsize=plt.rcParams['errorbar.capsize'])

    misc = Dictm(
        dpi=plt.rcParams['figure.dpi'],
        format=plt.rcParams['savefig.format'])

    return Dictm({'color': color, 'size': size, 'misc': misc})

# Figure preparation

def _calc_figsize(n_row: int, n_col: int, axsizeratio: Tuple[float, float]
                 ) -> Tuple[float]:
    """
    :py:data:`issych.typealias.Figure` に含まれる :py:data:`issych.typealias.Axes` の行数・列数に応じて、適切な :py:data:`issych.typealias.Figure` のサイズを計算します。

    Parameters
    ----------
    n_row: int
        ``Axes`` の行数です。
        一つの図において、縦に何個のプロットが並んでいるかです。
    n_col: int
        ``Axes`` の列数です。
        一つの図において、横に何個のプロットが並んでいるかです。
    axsizeratio: tuple
        X方向（横）の長さとY方向（縦）の長さの比率を指定してください。
        ``rcParams`` に設定している縦横比に、``Axes`` の行列数と ``axsizeratio``をかけたものが、返り値になります。

        **Examples**

        >>> axsizeratio=(2., 0.5)
        最終的に返される縦横の長さについて、横に2倍、縦に0.5倍されます。

    Returns
    -------
    ratio: tuple
        適切なX方向（横）の長さとY方向（縦）の長さです。
    """
    figsize = np.array(plt.rcParams['figure.figsize'])
    figsize *= (n_col, n_row)
    return tuple(figsize * axsizeratio)


def prepare_ax(axsizeratio: Tuple=(1., 1.),
               in_path_toml: Optional[Pathlike]=None) -> Tuple[Figure, Axes]:
    """
    一つの :py:data:`issych.typealias.Axes` を含む :py:data:`issych.typealias.Figure` を生成します。

    :py:func:`matplotlib.pyplot.subplot` とほぼ同一です。
    ただし、 :py:func:`set_rcparams` が事前に実行されます。

    Parameters
    ----------
    axsizeratio: tuple
        X方向（横）の長さとY方向（縦）の長さの比率を指定してください。
        ``rcParams`` に設定している縦横比にこの値をかけたサイズの ``Axes`` が生成されます。
    in_path_toml: 
        ここで指定された ``toml`` ファイルを読み取り、:py:data:`matplotlib.rcParams` に反映させます。
        ``toml`` ファイルには、 :ref:`issych-figure-setting-format` で記入されている必要があります。
    """
    if in_path_toml is None:
        set_rcparams()
    else:
        set_rcparams(in_path_toml)
    figsize = _calc_figsize(1, 1, axsizeratio)
    return plt.subplots(figsize=figsize)


def prepare_axes(n_row: int=1, n_col: int=1, axsizeratio: Tuple=(1., 1.),
                 in_path_toml: Optional[Pathlike]=None
                 ) -> Tuple[Figure, np.ndarray]:
    """
    指定した行数・列数の :py:data:`issych.typealias.Axes` を含む :py:data:`issych.typealias.Figure` を生成します。

    :py:func:`matplotlib.pyplot.subplots` とほぼ同一ですが、 :py:func:`set_rcparams` が事前に実行されます。
    また、 ``Axes`` が増えるとそれだけ図のサイズを大きくしてくれます。
    また、 ``n_row`` =1, ``n_col`` =1の場合でも、 ``Axes`` は :py:class:`numpy.ndarray` として返されます。

    Parameters
    ----------
    n_row: int
        ``Axes`` の行数です。
        一つの図において、縦に何個のプロットが並んでいるかです。
    n_col: int
        ``Axes`` の列数です。
        一つの図において、横に何個のプロットが並んでいるかです。
    axsizeratio: tuple
        X方向（横）の長さとY方向（縦）の長さの比率を指定してください。
        ``rcParams`` に設定している縦横比にこの値をかけたサイズの ``Axes`` が生成されます。
    in_path_toml: 
        ここで指定された ``toml`` ファイルを読み取り、:py:data:`matplotlib.rcParams` に反映させます。
        ``toml`` ファイルには、 :ref:`issych-figure-setting-format` で記入されている必要があります。 
    """
    if in_path_toml is None:
        set_rcparams()
    else:
        set_rcparams(in_path_toml)
    figsize = _calc_figsize(n_row, n_col, axsizeratio)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        return fig, np.array([axes])
    return fig, axes

# Low-level plotting

def draw_diag(ax, **kwargs) -> Axes:
    """
    図の対角線上に線が引かれます。

    Parameters
    ----------
    **kwargs:
        :py:func:`matplotlib.pyplot.plot` に渡す引数を指定できます。
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xlim, ylim, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


def mask_area(pos_from: float | pd.Timestamp, pos_to: float | pd.Timestamp,
              orient: Literal['horz', 'vert', 'h', 'v'], ax: Axes,
              color: Optional[str]=None, **kwargs) -> Axes:
    """
    指定した範囲を塗りつぶします。

    Parameters
    ----------
    pos_from: float or pd.Timestamp
        この位置以降の範囲を塗りつぶします。
    pos_to: float or pd.Timestamp
        この位置以前の範囲を塗りつぶします。
    orient: {'horz', 'vert', 'h', 'v'}
        'horz'または'h'を指定すると、指定されたX軸範囲を塗りつぶします。
        'vert'または'v'を指定すると、指定されたY軸範囲を塗りつぶします。
    ax: Axes
        描画する Axes です。
    color: str, optional
        塗りつぶすのに用いる色です。
    **kwargs:
        :py:func:`matplotlib.pyplot.add_patch` に渡す引数です。

    Examples
    --------
    >>> data = sns.load_dataset('fmri')
    >>> fig, ax = prepare_ax(in_path_toml=IN_PATH_TOML)
    >>> sns.lineplot(x="timepoint", y="signal", data=data, ax=ax)
    >>> mask_area(3.5, 7.5, orient='h', color='gray', ax=ax, zorder=-1)
    >>> mask_area(12.5, 12.5, orient='h', ax=ax, ec='white')
    >>> mask_area(0.08, 0.12, orient='vert', ax=ax, color='red')

    .. image:: /_static/test_mask_area.png
       :align: center
    """
    is_horz = orient.startswith('h')
    lim = ax.get_ylim() if is_horz else ax.get_xlim()
    xy = (pos_from, lim[0]) if is_horz else (lim[0], pos_from)
    size = pos_to - pos_from
    fulllength = lim[1] - lim[0]
    width = size if is_horz else fulllength
    height = fulllength if is_horz else size

    rect = Rectangle(xy=xy, width=width, height=height, fc=color, **kwargs)
    ax.add_patch(rect)
    return ax

# High-level plotting

def plot_within(dataset: pd.DataFrame,
                ax: Axes,
                x: str,
                y: str,
                ls: str='--',
                kwargs_scatter: Optional[dict]=None,
                kwargs_diagline: Optional[dict]=None) -> Axes:
    """
    いわゆる横持ち（Wide format）のデータフレームから、Within プロットを描画します。

    Parameters
    ----------
    dataset: pd.DataFrame
        図に用いるデータフレームです。
    ax: Axes
        描画する Axes です。
    x: str
        X軸に用いる列名です。
    y: str
        Y軸に用いる列名です。
    ls: str
        対角線上に引く線のラインスタイルです。
        :py:func:`matplotlib.pyplot.plot` に渡します。
    kwargs_scatter: dict, optional
        :py:func:`seaborn.scatterplot` に渡される引数です。
    kwargs_diagline: dict, optional
        :py:func:`matplotlib.pyplot.plot` に渡される引数です。

    Notes
    -----
    Within plot とは、対応のあるデータを正方形の散布図に描画し、対角線を引いただけのものです。
    たとえば、pre-post のデータを描画するのに便利です。
    ``x`` に pre、``y`` に post のデータを指定します。
    すると、対角線の上側にあるポイントは Post で値が増えていることになります。
    このように、各サンプルの変化を一目で確認できます。

    Examples
    --------
    >>> data = (sns.load_dataset('exercise')
    >>>         .pivot(index='id', columns='time', values='pulse'))
    >>> fig, ax = prepare_ax(in_path_toml=IN_PATH_TOML)
    >>> plot_within(data, ax=ax, x='1 min', y='30 min')

    .. image:: /_static/test_plot_within.png
       :align: center

    .. tip::
        Within plot という名前は Issych 作者が勝手につけたものです。
        正式な名称があったらどなたか教えてください。
    """
    kwargs_scatter = kwargs_scatter or dict()
    kwargs_diagline = kwargs_diagline or dict()
    color = get_current_rcparams().color

    fc = kwargs_scatter.pop('fc', color.main)
    ec = kwargs_scatter.pop('ec', color.main)
    color_line = kwargs_diagline.pop('color', color.sub)

    color = get_current_rcparams().color
    sns.scatterplot(data=dataset, x=x, y=y, ax=ax,
                    fc=fc, ec=ec, **kwargs_scatter)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    newlim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    ax.plot(newlim, newlim, **kwargs_diagline, ls=ls, color=color_line)
    ax.set_aspect('equal')
    ax.set_xlim(newlim)
    ax.set_ylim(newlim)

    return ax

def plot_raincloud(data: pd.DataFrame | Vector,
                   ax: Axes,
                   x: Optional[str | Vector]=None,
                   y: Optional[str | Vector]=None,
                   orient: Literal['h', 'v']='v',
                   strip: bool=True, box: bool=True, cloud: bool=True,
                   kwargs_strip: Optional[dict[str, Any]]=None,
                   kwargs_box: Optional[dict[str, Any]]=None,
                   kwargs_cloud: Optional[dict[str, Any]]=None) -> Axes:
    """
    RainCloud プロットを描画します。

    Parameters
    ----------
    dataset: pd.DataFrame
        図に用いるデータフレームです。
    ax: Axes
        描画する Axes です。
    x: str
        X軸に用いる列名です。
    y: str
        Y軸に用いる列名です。
    orient: {'h', 'v'}, default 'v'
        RainCloud プロットの方向です。
        `'h'` の場合は横方向、 `'v'` の場合は縦方向のプロットになります。
    strip: bool, default True
        Strip plot を描画するかどうかです。
    box: bool, default True
        箱ひげ図を描画するかどうかです。
    cloud: bool, default True
        半バイオリンプロットを描画するかどうかです。
    kwargs_strip: dict, optional
        :py:func:`seaborn.stripplot` に渡すキーワード引数です。
    kwargs_box: dict, optional
        :py:func:`seaborn.boxplot` に渡すキーワード引数です。
    kwargs_cloud: dict, optional
        :py:func:`seaborn.violinplot` に渡すキーワード引数です。

    Notes
    -----
    RainCloud プロットをサポートするパッケージとして、 ``PtitPrince`` が有名です。
    しかし ``PtitPrince`` は（2025年10月現在）しばらく更新が止まっています。
    ``PtitPrince`` を使うには、古い :mod:`pandas` をインストールする必要があります。
    本関数は ``PtitPrince`` ほど多様な機能はありませんが、最新の ``pandas`` をサポートします。

    References
    ----------
    PtitPrince: https://github.com/pog87/PtitPrince/tags

    Examples
    --------
    >>> data = sns.load_dataset('tips')
    >>> fig, axes = prepare_axes(3, 2, in_path_toml=IN_PATH_TOML)
    >>> plot_raincloud(data, ax=axes[0, 0], x='day', y='total_bill')
    >>> plot_raincloud(data.total_bill, ax=axes[1, 0],
    >>>                kwargs_strip={'color': 'red'},
    >>>                kwargs_box={'color': 'blue'})
    >>> plot_raincloud(data.total_bill, ax=axes[2, 0], box=False)
    >>> plot_raincloud(data.total_bill, ax=axes[0, 1], strip=False)
    >>> plot_raincloud(data.total_bill, ax=axes[1, 1], strip=False, box=False)

    .. image:: /_static/test_raincloud.png
       :align: center
    """
    def calc_jitter_width(ax: Axes):
        stripset = [child for child in ax.get_children()
                    if isinstance(child, PathCollection)]

        jitter_width = 0
        for strips in stripset:
            if orient =='v':
                offsets = strips.get_offsets()[:, 0]
            else:
                offsets = strips.get_offsets()[0, :]
            jitter_width = max(jitter_width, max(offsets) - min(offsets))

        return jitter_width

    def move_violin(violin: PathCollection, offset: float):
        pollypaths = violin.get_paths()
        for path in pollypaths:
            path.vertices[:, 0] -= offset

    ZORDER_LOW = 1
    ZORDER_HIGH = 2
    jitter_width = 0
    color = get_current_rcparams().color
    common_kwargs = {'data': data, 'x': x, 'y': y, 'ax': ax}

    kwargs_strip = kwargs_strip or dict()
    kwargs_box = kwargs_box or dict()
    kwargs_cloud = kwargs_cloud or dict()

    if strip:
        stripcolor = kwargs_strip.pop('color', color.main)
        stripzorder = kwargs_strip.pop('zorder', ZORDER_LOW)
        ax = sns.stripplot(orient=orient,
                           dodge=True,
                           color=stripcolor,
                           zorder=stripzorder,
                           **common_kwargs, **kwargs_strip)
        jitter_width = calc_jitter_width(ax)

    if box:
        DEFAULT_WIDTH = 0.2
        box_width = jitter_width if strip else DEFAULT_WIDTH
        fill = kwargs_box.pop('fill', False)
        boxcolor = kwargs_box.pop('color', color.highlight)
        flierprops = kwargs_box.pop('flierprops', {'marker': '_'})
        boxzorder = kwargs_box.pop('zorder', ZORDER_HIGH)
        width = kwargs_box.pop('width', box_width)
        ax = sns.boxplot(orient=orient,
                         fill=fill,
                         width=width,
                         color=boxcolor,
                         zorder=boxzorder,
                         flierprops=flierprops,
                         **common_kwargs, **kwargs_box)

    if cloud:
        DEFAULT_COEF_PADDING = 1.8
        DEFAULT_WIDTH = 0.3
        coef_padding = kwargs_cloud.pop('coef_pad', DEFAULT_COEF_PADDING)
        offset_size = coef_padding * jitter_width
        cloudcolor = kwargs_cloud.pop('color', color.sub)
        width = kwargs_cloud.pop('width', DEFAULT_WIDTH)
        ax = sns.violinplot(orient=orient,
                            hue_order=[True, False],
                            split=True,
                            color=cloudcolor,
                            inner=None,
                            width=width,
                            **common_kwargs, **kwargs_cloud)

        for halfviolin in ax.collections:
            move_violin(halfviolin, offset_size)

        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min - offset_size, x_max)

    return ax


class SigMarker:
    """
    棒グラフにマーカーを追加するためのクラスです。
    棒グラフは縦方向に伸びたものである必要があります。

    Examples
    --------
    >>> data = sns.load_dataset('tips')
    >>> fig, axes = prepare_axes(n_col=2, axsizeratio=(.5, 1.),
    >>>                          in_path_toml=IN_PATH_TOML)
    >>> sns.barplot(data=data, x='day', y='total_bill', hue='sex', ax=axes[0])
    >>> marker = SigMarker(axes[0])
    >>> marker.mark('patches', 0, 4, comment='1')
    >>> marker.mark('xticks', 0, 1, comment='2')
    >>> marker.mark('xticks', 0, 3, comment='3')
    >>> axes[0].legend_.remove()
    >>> custom_thrs = {1.1: 'n.s.', 0.01: '<0.01'}
    >>> sns.barplot(data=data, x='day', y='total_bill', hue='sex', ax=axes[1])
    >>> marker = SigMarker(axes[1], coef_interval_btw_layer=0.3)
    >>> marker.sigmark('xticks', 0, 2, p_value=0.50)
    >>> marker.sigmark('xticks', 1, 3, p_value=0.05)
    >>> marker.sigmark('patches', 3, 5, p_value=0.009)
    >>> marker.sigmark('patches', 6, 7, p_value=0.2, thresholds=custom_thrs)
    >>> marker.sigmark('patches', 5, 7, p_value=0.001, thresholds=custom_thrs)
    >>> axes[1].legend_.remove()

    .. image:: /_static/test_sigmarker.png
       :align: center
    """

    def __init__(self, ax: Axes,
                 coef_interval_btw_layer: float=.1,
                 coef_space_to_line: float=.1):
        """
        Parameters
        ----------
        ax: Axes
            描画する Axes です。
        coef_interval_btw_layer: float, default .1
            マーカー間の縦方向の距離です。
        coef_space_to_line: float, default .1
            マーカーとそれに対応するラインとの間の距離です。
        """
        self.layer: int = 0
        self.drawn_ranges: List[Tuple[float, float]] = []
        self.coef_interval = coef_interval_btw_layer
        self.coef_space_to_line = coef_space_to_line

        self.ax = ax
        self.ymax = ax.get_ylim()[1]
        containers = [container for container in ax.containers]  # type: ignore
        self.patches = [
            [contained] if isinstance(contained, Rectangle) else [*contained]
            for contained in containers]
        self.patches = sum([list(p) for p in np.array(self.patches).T], [])

    def _judge_conflict(self, x_pos: Tuple[float, float], drawn_ranges
                        ) -> bool:
        for rn in drawn_ranges:
            if (rn[0] <= x_pos[0] <= rn[1]) or (rn[0] <= x_pos[1] <= rn[1]):
                return True
        return False

    def mark(self, between: Literal['patches', 'xticks'],
             pos_from: int, pos_to: int, comment: str):
        """
        :py:class:`matplotlib.patches.Patch` （棒グラフの棒）または X軸目盛りの間にマーカーを描画します。

        Parameters
        ----------
        between: {'patches', 'xticks'}
            ``patches`` を指定した場合、あるバーの中心から別バーの中心の間に描画されます。
            ``xticks`` を指定した場合、X軸目盛りの間に描画されます。
        pos_from: int
            左から何番目（0始まり）の ``patch`` またはX軸目盛りからマーカー表示を始めるかです。
        pos_to: int
            左から何番目（0始まり）の ``patch`` またはX軸目盛りにまでマーカーを表示させるかです。
        comment: str
            表示させるマーカーの内容です。

        See also
        --------
        :py:meth:`sigmark`
        """
        def find_pos_bw_patches(patch_from: Rectangle, patch_to: Rectangle,
                                xpos: Tuple[Literal['c', 'r', 'l'],
                                            Literal['c', 'r', 'l']]=('c', 'c')
                                ) -> Tuple[float, float]:
            def find_xpos(patch, xpos):
                match xpos:
                    case 'l':
                        return patch.get_x()
                    case 'r':
                        return patch.get_x() + patch.get_width()
                    case 'c':
                        return patch.get_x() + (patch.get_width() / 2)

            xmin = find_xpos(patch_from, xpos[0])
            xmax = find_xpos(patch_to, xpos[1])

            return xmin, xmax

        def find_pos_bw_xticks(pos_from: int, pos_to: int, ax: Axes
                               ) -> Tuple[float, float]:
            xticks = ax.get_xticks()
            xmin = xticks[pos_from]
            xmax = xticks[pos_to]
            return xmin, xmax

        interval = self.coef_interval * (self.ymax - self.ax.get_ylim()[0])

        match between:
            case 'patches':
                x_from, x_to = find_pos_bw_patches(
                    self.patches[pos_from], self.patches[pos_to],  # type: ignore
                    xpos=('c', 'c'))
            case 'xticks':
                x_from, x_to = find_pos_bw_xticks(pos_from, pos_to, self.ax)

        if self._judge_conflict((x_from, x_to), self.drawn_ranges):
            self.layer += 1
            self.drawn_ranges = []
        self.drawn_ranges.append((x_from, x_to))

        ypos = self.ymax + (interval * self.layer)
        self.ax.hlines(y=ypos, xmin=x_from, xmax=x_to)
        self.ax.text((x_from + x_to) / 2,
                     ypos + (interval * self.coef_space_to_line),
                     comment, horizontalalignment='center')

    def sigmark(self, between: Literal['patches', 'xticks'],
                pos_from: int, pos_to: int, p_value: float,
                thresholds: Optional[dict]=None):
        """
        :py:meth:`SigMarker.mark` とほぼ同じですが、p値から自動的にマークを決定します。

        Parameters
        ----------
        thresholds: dict, optional
            p値の閾値と、その閾値を下回った際に表示すべき文字列を示す辞書です。
            デフォルトは、

            >>> {0.01: '**', 0.05: '*', 0.10: '†'}

            いずれの閾値も下回らない場合のマークを指定したい場合は、

            >>> {0.05: '*', 1.1: 'n.s.'}

            としてください。
            そういった指定がない場合は、''が返ります。
        
        See also
        --------
        :py:meth:`mark`
        """
        mark = Pvalue2SigMark(thresholds)(p_value)
        if len(mark) > 0:
            self.mark(between, pos_from, pos_to, mark)


def plot_corrmat(dataset: pd.DataFrame,
                 method: str='pearson',
                 in_path_toml: Pathlike='config/rcparams.toml',
                 thrs_p: Optional[Tuple[float, float, float]]=None,
                 sdgt: int=3,
                 rotation: int=30,
                 each_height: float=2.0,
                 abbr: Optional[Dictm]=None,
                 color_positive: Optional[str]=None,
                 color_negative: Optional[str]=None,
                 kwargs_violin: Optional[dict]=None,
                 kwargs_bar: Optional[dict]=None,
                 kwargs_regplot: Optional[dict]=None,
                 kwargs_circle: Optional[dict]=None) -> sns.axisgrid.PairGrid:
    """
    データフレームの各列間の相関関係を一目で確認するプロットを描画します。

    Parameters
    ----------
    dataset: pd.DataFrame
        プロットに用いるデータフレームです。
    method: {'pearson', 'spearman'}
        相関係数の算出に用いる手法です。
        :py:func:`pingouin.corr` に渡されますので、詳細はこちらを確認してください。
    in_path_toml: 
        ここで指定された ``toml`` ファイルを読み取り、:py:data:`matplotlib.rcParams` に反映させます。
        ``toml`` ファイルには、 `:ref:`issych-figure-setting-format` で記入されている必要があります。
    thrs_p: tuple, optional
        ``thrs_p[0]`` を下回る場合は薄い色の円が、``thrs_p[1]`` を下回る場合は濃い色の円が表示されます。
        デフォルトは、(.10, .05) です。
    sdgt: int, default 3
        表示する相関係数の、小数点以下桁数です。
    rotation: int, default 30
        ラベルの回転角度です。
    each_height: float, default 2.0
        図の大きさです。
    abbr: dict, optional
        略語をキー、正式語を値としたキーです。
        図に表示される変数名が、正式語に変換されます。
    color_positive: str, optional
        相関係数が正かつ、それに対応するp値が ``thrs_p`` で指定した値未満のときに用いる色です。
        デフォルトは、issych-figure-setting-format/color/highlight です。
    color_negative: str, optional
        相関係数が負かつ、それに対応するp値が ``thrs_p`` で指定した値未満のときに用いる色です。
        デフォルトは、issych-figure-setting-format/color/subhighlight です。
    kwargs_violin: dict, optional
        :py:func:`seaborn.violinplot` に渡す引数です。
    kwargs_bar: dict, optional
        :py:func:`seaborn.barplot` に渡す引数です。
    kwargs_regplot: dict, optional
        :py:func:`seaborn.regplot` に渡す引数です。
    kwargs_circle: dict, optional
        :py:func:`matplotlib.pyplot.Ellipse` に渡す引数です。
        相関係数の絶対値の大きさを示す円を表示するためのものです。

    Reference
    ---------
    https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another

    Examples
    --------
    >>> data = sns.load_dataset('iris')
    >>> data = pd.get_dummies(data, prefix='iris', dtype=int)
    >>> abbr = Dictm('tests/testdata/config/abbr.toml').general
    >>> g = plot_corrmat(data, method='spearman', sdgt=4, abbr=abbr,
    >>>                  color_positive='red', color_negative='blue',
    >>>                  in_path_toml='tests/testdata/config/rcparams.toml')

    .. image:: /_static/test_plot_corrmat2.png
       :align: center
    """
    def plot_corr(x: np.ndarray, y: np.ndarray, method: str, colors: Dictm,
                  color_posi: Optional[str], color_nega: Optional[str],
                  thrs_p: Optional[Tuple[float, float, float]],
                  sdgt: int, **_):
        def select_circle_color(p: float, thrs_p: Tuple[float, float],
                                coef: float, color_posi: Optional[str],
                                color_nega: Optional[str]
                                ) -> Tuple[str, float]:
            if p >= thrs_p[0]:
                return colors.sub, 0.2
            color_posi = color_posi or colors.highlight
            color_nega = color_nega or colors.subhighlight
            color_circle = color_posi if coef >= 0 else color_nega
            alpha = 0.5 if p >= thrs_p[1] else 1.0
            return color_circle, alpha

        if thrs_p is None:
            thrs_p = (.10, .05)
        coef, p = pg.corr(x, y, method=method)[['r', 'p-val']].values[0]

        circle_color, circle_alpha = select_circle_color(
            p, thrs_p, coef, color_posi, color_nega)

        ax = plt.gca()
        center_x = np.sum(ax.get_xlim()) / 2.
        center_y = np.sum(ax.get_ylim()) / 2.
        width = np.abs(np.diff(ax.get_xlim()) * coef)[0]
        height = np.abs(np.diff(ax.get_ylim()) * coef)[0]

        coef_text = f'{str(coef.round(sdgt)).replace("0.", ".")}'
        n_shortage = sdgt - (len(coef_text) - coef_text.find('.') - 1)
        coef_text = coef_text + ('0' * n_shortage)

        ax.add_patch(Ellipse(xy=(center_x, center_y), width=width,
                             height=height, fc=circle_color,
                             alpha=circle_alpha, linewidth=0.))
        ax.annotate(coef_text, [.5, .5,],  xycoords="axes fraction",
                    ha='center', va='center', color=colors.main)
        ax.set_axis_off()

    def plot_dist(data: np.ndarray, kwargs_violin: dict, kwargs_bar: dict,
                  **_):
        colors = get_current_rcparams().color
        color_violin = kwargs_violin.pop('color', colors.main)
        lw_violin = kwargs_violin.pop('lw', 0)

        color_bar = kwargs_bar.pop('color', colors.main)
        if len(data.unique()) <= 2:
            sns.barplot(data.value_counts(), color=color_bar, **kwargs_bar)
        else:
            sns.violinplot(x=data, color=color_violin, linewidth=lw_violin,
                           **kwargs_violin)

    set_rcparams(in_path_toml)
    colors = get_current_rcparams().color

    kwargs_regplot = kwargs_regplot or dict(scatter_kws=dict())
    color_regplot = kwargs_regplot.pop('color', colors.main)
    ec_scatter = kwargs_regplot['scatter_kws'].pop('edgecolor', 'none')
    kwargs_regplot['scatter_kws']['edgecolor'] = ec_scatter

    kwargs_violin = kwargs_violin or dict()
    kwargs_bar = kwargs_bar or dict()

    kwargs_circle = kwargs_circle or dict()

    dataset = dataset.copy().select_dtypes([int, float])
    g = sns.PairGrid(dataset, diag_sharey=False, height=each_height)
    g.map_lower(sns.regplot, color=color_regplot, **kwargs_regplot)
    g.map_diag(plot_dist, kwargs_violin=kwargs_violin, kwargs_bar=kwargs_bar)
    g.map_upper(plot_corr, method=method, colors=colors,
                color_posi=color_positive, color_nega=color_negative,
                thrs_p=thrs_p, sdgt=sdgt, **kwargs_circle)
    if abbr:
        for ax in g.axes.flatten():
            ax.set_xlabel(abbr.may(ax.get_xlabel()), rotation=rotation)
            ax.set_ylabel(abbr.may(ax.get_ylabel()), rotation=0., ha='right')
    return g
