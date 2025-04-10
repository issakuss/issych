from typing import TypeAlias, Sequence
from pathlib import Path

import pandas as pd

from matplotlib import figure, axes


Number: TypeAlias = int | float
Vector: TypeAlias = Sequence[Number] | pd.Series
Pathlike: TypeAlias = Path | str
Figure: TypeAlias = figure.Figure  # plt.Figure is not recognized by pylance
Axes: TypeAlias = axes.Axes | axes._base._AxesBase
