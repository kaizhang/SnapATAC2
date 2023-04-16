import snapatac2 as snap
from pathlib import Path
import numpy as np

def h5ad(dir=Path("./")):
    import uuid
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

'''
def test_exclude():
    fragment_file = snap.datasets.pbmc500(True)

    chr_sizes = snap.genome.hg38.chrom_sizes
    chr_sizes.pop('chr1', None)
    chr_sizes.pop('chr10', None)
    data1 = snap.pp.import_data(
        fragment_file,
        genome=snap.genome.hg38,
        chrom_size=chr_sizes,
        sorted_by_barcode=False
    )
    snap.pp.add_tile_matrix(data1, exclude_chroms=None)

    data2 = snap.pp.import_data(fragment_file, genome=snap.genome.hg38, sorted_by_barcode=False)
    snap.pp.add_tile_matrix(data2, exclude_chroms=["chr1", "chr10"])

    np.testing.assert_array_equal(data1.X.data, data2.X.data)
    np.testing.assert_array_equal(data1.obs_names, data2.obs_names)
    np.testing.assert_array_equal(data1.var_names, data2.var_names)


def test_backed(tmp_path):
    fragment_file = snap.datasets.pbmc500(True)

    data = snap.pp.import_data(
        fragment_file,
        genome=snap.genome.hg38,
        file=h5ad(tmp_path),
        sorted_by_barcode=False,
    )
    snap.pp.add_tile_matrix(data)

    snap.pp.filter_cells(data)
    snap.pp.select_features(data)

    snap.tl.spectral(data, sample_size=100)
    snap.tl.spectral(data)
    snap.pp.knn(data)
    snap.tl.leiden(data)

    snap.pp.make_gene_matrix(data, gene_anno=snap.genome.hg38, file = h5ad(tmp_path))

def test_in_memory():
    fragment_file = snap.datasets.pbmc500(True)

    data = snap.pp.import_data(
        fragment_file,
        genome=snap.genome.hg38,
        sorted_by_barcode=False,
    )
    snap.pp.add_tile_matrix(data)

    snap.pp.filter_cells(data)
    snap.pp.select_features(data)

    snap.tl.spectral(data, sample_size=100)
    snap.tl.spectral(data)
    snap.pp.knn(data)
    snap.tl.leiden(data)

    snap.pp.make_gene_matrix(data, gene_anno=snap.genome.hg38)
'''