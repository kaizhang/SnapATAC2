from ._io import read_10x_mtx
from . import preprocessing as pp
from . import tools as tl
from . import plotting as pl
from . import export as ex

from snapatac2._snapatac2 import (
    AnnData, AnnDataSet, read, read_mtx, create_dataset, read_dataset,
    LinkData, NodeData,
)

import sys
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp', 'tl', 'pl', 'ex']})
del sys