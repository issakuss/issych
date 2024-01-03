from typing import Optional, Any, Literal, Tuple, List
from pathlib import Path

from cycler import cycler
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle, Ellipse
import seaborn as sns

from .dataclass import Dictm
from .typealias import Pathlike, Figure, Axes
from .fileio import load_config
from .stat import Pvalue2SigMark


DEFAULT_PATH_RCPARAMS = 'config/rcparams.ini'

# Settings

def load_rcparams(path_ini :Optional[Pathlike]=None):
    path_ini = path_ini if path_ini else DEFAULT_PATH_RCPARAMS
    color = load_config(path_ini).color
    size = load_config(path_ini).size
    misc = load_config(path_ini).misc
    return color, size, misc


def set_rcparams(path_ini :Optional[Pathlike]=None):
    color_, size_, misc_ = get_rcparams()
    color, size, misc = load_rcparams(path_ini)

    plt.rcParams.update(
        {'axes.spines.right': False,
         'axes.spines.top': False,
         'errorbar.capsize': size.capsize,
         'figure.dpi': misc.dpi,
         'figure.figsize': size.figsize,
         'font.size': size.fontsize,
         'savefig.format': misc.format,
         'axes.edgecolor': color.main,
         'axes.facecolor': color.back,
         'axes.labelcolor': color.main,
         'axes.prop_cycle': cycler(color=color.cycle),
         'figure.edgecolor': color.main,
         'figure.facecolor': color.back,
         'ytick.color': color.main,
         'xtick.color': color.main,
         'savefig.facecolor': color.back,
         'lines.color': color.main,
         'lines.linewidth': size.linewidth,
         'text.color': color.main,
         'patch.facecolor': color.highlight,
         'patch.edgecolor': color.subhighlight or color_.subhighlight,
         'grid.color': color.sub})


def get_rcparams() -> Tuple[Dictm, Dictm, Dictm]:
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
    
    return color, size, misc

# Figure preparation

def calc_figsize(n_row: int, n_col: int, axsizeratio: Tuple) -> Tuple[float]:
    figsize = np.array(plt.rcParams['figure.figsize'])
    figsize *= (n_col, n_row)
    return tuple(figsize * axsizeratio)


def prepare_ax(axsizeratio: Tuple=(1., 1.), path_ini: Optional[Pathlike]=None
               ) -> Tuple[Figure, Axes]:
    """
    Tested in test_plot_raincloud.py
    """
    set_rcparams(path_ini)
    figsize = calc_figsize(1, 1, axsizeratio)
    return plt.subplots(figsize=figsize)


def prepare_axes(n_row: int=1, n_col: int=1, axsizeratio: Tuple=(1., 1.),
                 path_ini: Optional[Pathlike]=None
                 ) -> Tuple[Figure, np.ndarray]:
    """
    Tested in test_SigMarker.py
    """
    set_rcparams(path_ini)
    figsize = calc_figsize(n_row, n_col, axsizeratio)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        return fig, np.array([axes])
    return fig, axes

# Plotting

def draw_line(pos: float, orient: Literal['horz', 'vert', 'h', 'v'],
              ax: Axes, **kwargs):
    """
    Use ax.axvline() and ax.axhline() instead. 
    """
    vert = (orient in ['vert', 'v'])
    lim = ax.get_ylim() if vert else ax.get_xlim()
    func = ax.vlines if vert else ax.hlines

    func(pos, *lim, **kwargs)


def draw_diag(ax, **kwargs):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xlim, ylim, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def mask_area(pos_from: float | pd.Timestamp, pos_to: float | pd.Timestamp,
              orient: Literal['horz', 'vert', 'h', 'v'], ax: Axes,
              color: Optional[str]=None, **kwargs) -> Axes:
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


def plot_halfviolin(data: pd.DataFrame, ax: Axes,
                    x: Optional[str]=None, y: Optional[str]=None,
                    color: Optional[str]=None, xdodge_size: int=200,
                    coef_xlim: float=1/1000) -> Axes:
    """
    Plot half-violin without Ptitprince.
    Tested in test_plot_raincloud.py
    """
    n = len(data)
    data = pd.concat([data] * 2).reset_index(drop=True)
    data['__dummy_col'] = (['dummy_a'] * n) + (['dummy_b'] * n)

    # sns.violinplot disables hue when x or y is None
    dummy_axis = 'dummy_axis'
    data[dummy_axis] = 'dummy'
    x = x or dummy_axis
    y = y or dummy_axis

    frontcolor = color or plt.rcParams['patch.facecolor']
    sns.violinplot(data=data, x=x, y=y, split=True, color=frontcolor,
                   linewidth=0., ax=ax)

    for collection in ax.collections:  # type: ignore
        if not isinstance(collection, PolyCollection):
            continue
        offsets = collection.get_offsets()
        offsets[:, 0] -= xdodge_size  # type: ignore
        collection.set_offsets(offsets)
    ax.set_xlim(ax.get_xlim()[0] - (xdodge_size * coef_xlim), ax.get_xlim()[1])

    return ax


def plot_raincloud(data: pd.DataFrame, ax: Axes,
                   x: Optional[str]=None, y: Optional[str]=None,
                   strip: bool=True, stripkwargs: dict[str, Any]=dict(),
                   cloud: bool=True, cloudkwargs: dict[str, Any]=dict(),
                   box: bool=True, boxkwargs: dict[str, Any]=dict()
                   ) -> Axes:
    """
    Not support hue.
    """

    color, *_ = get_rcparams()
    kwargs = {'x': x, 'y': y, 'data': data, 'ax': ax}

    ZORDER_LOW = 1
    ZORDER_HIGH = 2

    if strip:
        stripcolor = stripkwargs.pop('color', color.main)
        stripzorder = stripkwargs.pop('zorder', ZORDER_LOW)
        ax = sns.stripplot(dodge=True, color=stripcolor, zorder=stripzorder,
                           **kwargs, **stripkwargs)

    if box:
        WIDTH = 0.2
        fill = boxkwargs.pop('fill', False)
        boxcolor = boxkwargs.pop('color', color.highlight)
        flierprops = boxkwargs.pop('flierprops', {'marker': '_'})
        boxzorder = boxkwargs.pop('zorder', ZORDER_HIGH)
        width = boxkwargs.pop('width', WIDTH)
        ax = sns.boxplot(fill=fill, width=width, color=boxcolor,
                         zorder=boxzorder, flierprops=flierprops, **kwargs,
                         **boxkwargs)

    if cloud:
        cloudcolor = cloudkwargs.pop('color', color.sub)
        ax = plot_halfviolin(color=cloudcolor, **kwargs, **cloudkwargs)

    return ax


def _find_pos_bw_patches(patch_from: Rectangle, patch_to: Rectangle,
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


def _find_pos_bw_xticks(pos_from: int, pos_to: int, ax: Axes
                        ) -> Tuple[float, float]:
    xticks = ax.get_xticks()
    xmin = xticks[pos_from]
    xmax = xticks[pos_to]
    return xmin, xmax


class SigMarker:
    """
    Mark significance between barplots.
    """
    def __init__(self, ax: Axes):
        self.layer: int = 0
        self.drawn_ranges: List[Tuple[float, float]] = []
        self.INTERVAL: float = .1

        self.ax = ax
        self.ymax = ax.get_ylim()[1]
        containers = [container for container in ax.containers]  # type: ignore
        self.patches = [
            [contained] if isinstance(contained, Rectangle) else [*contained]
            for contained in containers]
        self.patches = sum([list(p) for p in np.array(self.patches).T], [])

    def judge_conflict(self, x_pos: Tuple[float, float], drawn_ranges) -> bool:
        for rn in drawn_ranges:
            if (rn[0] <= x_pos[0] <= rn[1]) or (rn[0] <= x_pos[1] <= rn[1]):
                return True
        return False

    def mark(self, between: Literal['patches', 'xticks'],
             pos_from: int, pos_to: int, comment: str):
        interval = self.INTERVAL * (self.ymax - self.ax.get_ylim()[0])
        
        match between:
            case 'patches':
                x_from, x_to = _find_pos_bw_patches(
                    self.patches[pos_from], self.patches[pos_to],  # type: ignore
                    xpos=('c', 'c'))
            case 'xticks':
                x_from, x_to = _find_pos_bw_xticks(pos_from, pos_to, self.ax)
                    
        if self.judge_conflict((x_from, x_to), self.drawn_ranges):
            self.layer += 1
            self.drawn_ranges = []
        self.drawn_ranges.append((x_from, x_to))

        ypos = self.ymax + (interval * self.layer)
        self.ax.hlines(y=ypos, xmin=x_from, xmax=x_to)
        self.ax.text((x_from + x_to) / 2, ypos, comment,
                     horizontalalignment='center')

    def sigmark(self, between: Literal['patches', 'xticks'],
                pos_from: int, pos_to: int, p_value: float,
                thresholds: Optional[dict]=None, ns_comment: str=''):
        mark = Pvalue2SigMark(thresholds, ns_comment)(p_value)
        if len(mark) > 0:
            self.mark(between, pos_from, pos_to, mark)


def plot_corrmat(dataset: pd.DataFrame, method: str='pearson',
                 path_ini: Optional[Pathlike]=None,
                 thrs_p: Optional[Tuple[float, float, float]]=None,
                 rotation: int=30, each_height: float=2.0,
                 abbr: Optional[Dictm]=None, kwargs_diag: dict={},
                 kwargs_lower: dict={}, kwargs_upper: dict={}
                 ) -> sns.axisgrid.PairGrid:
    # https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another
    def plot_corr(x: np.ndarray, y: np.ndarray, method: str, color: Dictm,
                  thrs_p: Optional[Tuple[float, float, float]]=None, **kwargs):
        FONTSIZE = 32.
        SDGT = 3
        if thrs_p is None:
            thrs_p = (.10, .05, .01)
        coef, p = pg.corr(x, y, method=method)[['r', 'p-val']].values[0]

        c = color.sub
        idx_color = 0 if coef >= 0 else 1
        c = color.cycle[idx_color] if p < thrs_p[0] else c
        alpha = 0.2
        alpha = 0.5 if p < thrs_p[-2] else alpha
        alpha = 1.0 if p < thrs_p[-1] else alpha
        
        ax = plt.gca()
        center_x = np.sum(ax.get_xlim()) / 2.
        center_y = np.sum(ax.get_ylim()) / 2.
        width = np.abs(np.diff(ax.get_xlim()) * coef)[0]
        height = np.abs(np.diff(ax.get_ylim()) * coef)[0]

        coef_text = f'{str(coef.round(3)).replace("0.", ".")}'
        n_shortage = SDGT - (len(coef_text) - coef_text.find('.') - 1)
        coef_text = coef_text + ('0' * n_shortage)

        ax.add_patch(Ellipse(
            xy=(center_x, center_y), width=width, height=height, fc=c,
            alpha=alpha, linewidth=0.))
        ax.annotate(coef_text, [.5, .5,],  xycoords="axes fraction",
                    ha='center', va='center', fontsize=FONTSIZE,
                    color=color.main)
        ax.set_axis_off()
    
    def plot_dist(data: np.ndarray, color: str, lw: float, **kwargs):
        if len(data.unique()) <= 2:
            sns.barplot(data.value_counts(), color=color)
        else:
            sns.violinplot(x=data, linewidth=lw, color=color, inner='box')

    path_ini = path_ini if path_ini else DEFAULT_PATH_RCPARAMS
    abbr = abbr if abbr else load_config('config/abbr.ini').flatten()

    color, *_ = load_rcparams(path_ini)
    set_rcparams(path_ini)

    dataset = dataset.copy().select_dtypes([int, float])
    g = sns.PairGrid(dataset, diag_sharey=False, height=each_height)
    g.map_lower(sns.regplot, color=color.sub, **kwargs_lower)
    g.map_diag(plot_dist, color=color.sub, **kwargs_diag)
    g.map_upper(plot_corr, method=method, color=color, thrs_p=thrs_p,
                **kwargs_upper)
    for ax in g.axes.flatten():
        if abbr:
            ax.set_xlabel(abbr.full(ax.get_xlabel()), rotation=rotation)
            ax.set_ylabel(abbr.full(ax.get_ylabel()), rotation=0., ha='right')
    return g