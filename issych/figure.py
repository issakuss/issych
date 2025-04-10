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
    in_path_tomlに指定されたtomlファイルにある設定を反映させます。
    tomlファイルには、issych-figure-setting-formatで記入されている必要があります。
    issych-figure-setting-formatの例は以下の通りです。

    [color]
    main = '#EAE6E5'          # 主に用いられる色；黒など
    back = '#011627'          # 背景色；白など
    sub = '#3C4A57'           # mainで指定した色とは別の色；灰色など
    highlight = '#8FCB9B'     # 強調するための色；赤など
    subhighlight = '#EF6F6C'  # highlightで指定した色とは別の強調色；青など
    cycle = [                 # 複数の色が用いられる場合のパターン
    '#E69F00',                # 一つ目の色） 
    '#56B4E9',                # 二つ目の色）
    '#009E73',                # …色数は自由）
    '#F0E442',
    '#0072B2',
    '#D55E00',
    '#CC79A7'
    ]

    [size]
    figsize = [8.0, 4.0]      # [横幅, 縦幅] を表す数値
    fontsize = 16.0           # フォントサイズを表す数値
    linewidth = 1.0           # 線の太さを表す数値
    capsize = 1.0             # エラーバーのキャップの長さを表す数値

    [misc]
    dpi = 256.0               # 解像度を表す数値
    format = 'eps'            # 保存する際のフォーマットを表す文字列；pngなど

    Tested with: prepare_ax(), prepare_axes()
    """
    params = Dynaconf(settings_files=in_path_toml)

    plt.rcParams.update(
        {'axes.spines.right': False,
         'axes.spines.top': False,
         'errorbar.capsize': params.size.capsize,
         'figure.dpi': params.misc.dpi,
         'figure.figsize': params.size.figsize,
         'font.size': params.size.fontsize,
         'savefig.format': params.misc.format,
         'axes.edgecolor': params.color.main,
         'axes.facecolor': params.color.back,
         'axes.labelcolor': params.color.main,
         'axes.prop_cycle': cycler(color=params.color.cycle),
         'figure.edgecolor': params.color.main,
         'figure.facecolor': params.color.back,
         'ytick.color': params.color.main,
         'xtick.color': params.color.main,
         'savefig.facecolor': params.color.back,
         'lines.color': params.color.main,
         'lines.linewidth': params.size.linewidth,
         'text.color': params.color.main,
         'patch.facecolor': params.color.highlight,
         'patch.edgecolor': params.color.subhighlight,
         'grid.color': params.color.sub})


def get_current_rcparams() -> Tuple[Dictm, Dictm, Dictm]:
    """
    現在のrcparamsを、issych-figure-setting-formatで取得します。

    Tested with: plot_prepost(), plot_raincloud()
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

def calc_figsize(n_row: int, n_col: int, axsizeratio: Tuple) -> Tuple[float]:
    """
    Figureに含まれるAxesの行数・列数に応じて、適切なFigureのサイズを計算します。

    Tested with: prepare_axes()
    """
    figsize = np.array(plt.rcParams['figure.figsize'])
    figsize *= (n_col, n_row)
    return tuple(figsize * axsizeratio)


def prepare_ax(axsizeratio: Tuple=(1., 1.),
               in_path_toml: Optional[Pathlike]=None) -> Tuple[Figure, Axes]:
    """
    ひとつのAxesを含むFigureを生成します。
    plt.subplot()とほぼ同一ですが、set_rcparams()が事前に実行されます。

    Tested with: TestMaskArea(), TestPlotPrePost()
    """
    set_rcparams(in_path_toml)
    figsize = calc_figsize(1, 1, axsizeratio)
    return plt.subplots(figsize=figsize)


def prepare_axes(n_row: int=1, n_col: int=1, axsizeratio: Tuple=(1., 1.),
                 in_path_toml: Optional[Pathlike]=None
                 ) -> Tuple[Figure, np.ndarray]:
    """
    指定した行数・列数のAxesを含むFigureを生成します。
    plt.subplots()とほぼ同一ですが、set_rcparams()が事前に実行されます。
    また、n_row=1, n_col=1の場合でも、Axesはnp.ndarrayとして返されます。

    Tested with: TestPlotRainCloud()
    """
    set_rcparams(in_path_toml)
    figsize = calc_figsize(n_row, n_col, axsizeratio)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        return fig, np.array([axes])
    return fig, axes

# Low-level plotting

def draw_diag(ax, **kwargs) -> Axes:
    """
    対角線上に線が引かれます。
    kwargsには、ax.plot()に渡す引数を指定できます。

    Tested with: plot_prepost()
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
    pos_fromからpos_toまでの範囲を塗りつぶします。
    orientに'horz'または'h'を指定すると、指定されたX軸範囲を塗りつぶします。
    'vert'または'v'を指定すると、指定されたY軸範囲を塗りつぶします。
    kwargsには、ax.add_patch()に渡す引数を指定できます。
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
    Wide-formatのdataから、Within plotを描画します。
    Within plotは、対応のあるデータを正方形の散布図に描画し、対角線を引いただけのものです
    （名前はissych作者が勝手につけました）。
    たとえば、pre-postのデータを描画するのに便利です。
    xにpre、yにpostのデータを指定します。
    すると、対角線の上側にあるポイントはPostで値が増えていることになります。
    各サンプルの変化を一目で確認できます。
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
    RainCloudプロットを描画します。
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
    """
    def __init__(self, ax: Axes,
                 coef_interval_btw_layer: float=.1,
                 coef_space_to_line: float=.1):
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
        Patchまたはxticksの間にマーカーを描画します。
        betweenにpatchesを指定した場合、あるバーの中心から別バーの中心の間に描画されます。
        xticksを指定した場合、xticksの間に描画されます。
        pos_fromとpos_toには、左から何番目（0始まり）のpatchまたはxticksかを指定します。
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
        mark()とほぼ同じですが、p値から自動的にマークを決定します。
        thresholds: dict
            p値の閾値と、その閾値を下回った際に表示すべき文字列を示す辞書。
            デフォルトは、
            >> {0.01: '**', 0.05: '*', 0.10: '†'}
            いずれの閾値も下回らない場合のマークを指定したい場合は、
            >> {0.05: '*', 1.1: 'n.s.'}
            としてください。
            そういった指定がない場合は、''が返ります。
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
    datasetの各列間の相関関係を一目で確認するプロットを描画します。
    thrs_p[0]を下回る場合は薄い色の円が、thrs_p[1]を下回る場合は濃い色の円が表示されます。
    参考：
    https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another
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
