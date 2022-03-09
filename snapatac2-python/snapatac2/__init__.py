from ._io import read_10x_mtx
from . import preprocessing as pp
from . import tools as tl
from . import genome
from . import plotting as pl
from ._utils import create_ann_collection

from anndata import AnnData, concat, read, read_h5ad, read_mtx

import snapatac2._file_specs

import sys
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp', 'tl', 'pl']})
del sys