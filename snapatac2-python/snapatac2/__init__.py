from . import preprocessing as pp
from . import tools as tl
from . import genome
from . import plotting as pl

from anndata import AnnData, concat, read, read_h5ad, read_mtx
from anndata.experimental import AnnCollection

import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp', 'tl']})

del sys