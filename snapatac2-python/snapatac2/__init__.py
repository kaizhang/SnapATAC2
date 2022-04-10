#from ._io import read_10x_mtx
from . import preprocessing as pp
from . import tools as tl
from . import genome
from . import plotting as pl

from snapatac2._snapatac2 import AnnData, AnnDataSet, read, read_mtx, create_dataset, read_dataset

import sys
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp', 'tl', 'pl']})
del sys