from typing import Optional, Any, Tuple, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import seaborn as sns

from .typealias import Pathlike, Figure, Axes
from .fileio import load_config


DEFAULT_PATH_RCPARAMS = Path(__file__).parents[2] / 'config/rcparams.ini'

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
    set_rcparams(path_ini)
    figsize = calc_figsize(n_row, n_col, axsizeratio, path_ini)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        return fig, np.array([axes])
    return fig, axes

# Plotting

def draw_line(pos: float, orient: Literal['horz', 'vert', 'h', 'v'],
              ax: plt.Axes, **kwargs):
    vert = (orient in ['vert', 'v'])
    lim = ax.get_ylim() if vert else ax.get_xlim()
    func = ax.vlines if vert else ax.hlines

    func(pos, *lim, **kwargs)


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
                   xcol: Optional[None]=None, ycol: Optional[str]=None,
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
