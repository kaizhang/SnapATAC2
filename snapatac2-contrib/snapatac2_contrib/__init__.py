from ._version import __version__
from . import metrics

import sys

import logging
logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO, 
)

del sys