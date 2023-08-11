import snapatac2 as snap
from pathlib import Path
import numpy as np

def h5ad(dir=Path("./")):
    import uuid
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

def test_in_memory():
    fragment_file = snap.datasets.pbmc500(True)

    data = snap.pp.import_data(
        fragment_file,
        genome=snap.genome.hg38,
        sorted_by_barcode=False,
    )

    snap.pp.add_tile_matrix(data)