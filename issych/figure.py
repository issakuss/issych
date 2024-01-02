from typing import Optional, Any, Literal, Tuple, List

from cycler import cycler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
import seaborn as sns

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
         'grid.color': color.sub})

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
                    xcol: Optional[str]=None, ycol: Optional[str]=None,
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
    xcol = xcol or dummy_axis
    ycol = ycol or dummy_axis

    frontcolor = color or plt.rcParams['patch.facecolor']
    backcolor = plt.rcParams['axes.facecolor']
    palette = {'dummy_a': frontcolor, 'dummy_b': backcolor}

    sns.violinplot(data=data, x=xcol, y=ycol, hue='__dummy_col',
                   split=True, palette=palette, linewidth=0., ax=ax)
    ax.get_legend().remove()  # type: ignore
    if xcol == dummy_axis:
        ax.set_xlabel('')
        ax.set_xticklabels('')
    if ycol == dummy_axis:
        ax.set_ylabel('')
        ax.set_yticklabels('')

    for collection in ax.collections:  # type: ignore
        if not isinstance(collection, PolyCollection):
            continue
        offsets = collection.get_offsets()
        offsets[:, 0] -= xdodge_size  # type: ignore
        collection.set_offsets(offsets)
    ax.set_xlim(ax.get_xlim()[0] - (xdodge_size * coef_xlim), ax.get_xlim()[1])

    return ax


def plot_raincloud(data: pd.DataFrame, ax: Axes,
                   xcol: Optional[str]=None, ycol: Optional[str]=None,
                   strip: bool=True, stripkwargs: dict[str, Any]=dict(),
                   cloud: bool=True, cloudkwargs: dict[str, Any]=dict()
                   ) -> Axes:
    """
    Not support hue.
    """

    color_main = plt.rcParams['axes.edgecolor']
    color_sub = plt.rcParams['grid.color']

    if strip:
        stripcolor = stripkwargs.pop('color', None) or color_main
        ax = sns.stripplot(data, x=xcol, y=ycol, dodge=True,
                           color=stripcolor, ax=ax, **stripkwargs)

    if cloud:
        cloudcolor = cloudkwargs.pop('color', None) or color_sub
        ax = plot_halfviolin(
            data, ax, xcol, ycol, color=cloudcolor, **cloudkwargs)

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

