from typing import TypeAlias
from pathlib import Path

from matplotlib import figure, axes


Pathlike: TypeAlias = Path | str
Figure: TypeAlias = figure.Figure  # plt.Figure is not recognized by pylance
Axes: TypeAlias = axes.Axes
