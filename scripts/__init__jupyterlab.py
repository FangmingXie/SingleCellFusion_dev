"""Import commonly used libraries"""

import numpy as np
import pandas as pd
import collections
from natsort import natsorted

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42 # editable text in matplotlib
mpl.rcParams['svg.fonttype'] = 'none'

import matplotlib.ticker as mtick
PercentFormat = mtick.FuncFormatter(lambda y, _: '{:.3%}'.format(y))
ScalarFormat = mtick.ScalarFormatter()


# seaborn
import seaborn as sns
sns.set_style('ticks', rc={'axes.grid':True})
sns.set_context('talk')

# set matplotlib formats
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')







# # For every axis, set the x and y major locator
# for axi in ax.flat:
#     axi.xaxis.set_major_locator(plt.MaxNLocator(3))
#     axi.yaxis.set_major_locator(plt.MaxNLocator(3))
# fig

# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

