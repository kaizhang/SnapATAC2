from ._version import __version__
from ._io import read_10x_mtx
from . import preprocessing as pp
from . import tools as tl
from . import plotting as pl
from . import export as ex

from snapatac2._snapatac2 import (
    AnnData, AnnDataSet, PyDNAMotif,
    read, read_mtx, read_csv, read_dataset, read_motifs,
)

import sys
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp', 'tl', 'pl', 'ex']})

import logging
logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO, 
)

del sys