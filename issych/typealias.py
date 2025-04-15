"""型エイリアス定義モジュールです。"""

from typing import TypeAlias, Sequence
from pathlib import Path

import pandas as pd

from matplotlib import figure, axes


#: 数値を表す型です。int or float
Number: TypeAlias = int | float

#: ベクトルを表す型です。Sequence or :py:class:`pandas.Series`
Vector: TypeAlias = Sequence[Number] | pd.Series

#: ファイルパスを表す型です。:py:class:`pathlib.Path` or str
Pathlike: TypeAlias = Path | str

#: ``matplotlib`` の図を示す型です
Figure: TypeAlias = figure.Figure  # plt.Figure is not recognized by pylance

#: ``matplotlib`` の軸を示す型です
Axes: TypeAlias = axes.Axes | axes._base._AxesBase
