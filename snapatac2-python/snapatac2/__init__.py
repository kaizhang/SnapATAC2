from . import preprocessing as pp
from . import tools as tl
from . import genome

from anndata import AnnData, concat, read, read_h5ad
from anndata.experimental import AnnCollection

import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp', 'tl']})

del sys