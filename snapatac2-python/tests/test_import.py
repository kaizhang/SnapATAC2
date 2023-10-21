import snapatac2 as snap
from pathlib import Path
import numpy as np
import gzip

def h5ad(dir=Path("./")):
    import uuid
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

def read_bed(bed_file):
    with gzip.open(bed_file, 'rt') as f:
        return sorted([line.strip().split('\t')[:4] for line in f if line.startswith('chr')])

def test_in_memory():
    fragment_file = snap.datasets.pbmc500(downsample=True)

    data = snap.pp.import_data(
        fragment_file,
        chrom_sizes=snap.genome.hg38,
        min_num_fragments=0,
        sorted_by_barcode=False,
    )

    data.obs['group'] = '1'
    snap.ex.export_bed(data, groupby="group", suffix='.bed.gz')

    assert read_bed("1.bed.gz") == read_bed(fragment_file)