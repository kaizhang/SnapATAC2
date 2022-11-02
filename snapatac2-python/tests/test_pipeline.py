import snapatac2 as snap
from pathlib import Path

def h5ad(dir=Path("./")):
    import uuid
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

def test_pipeline(tmp_path):
    fragment_file = snap.datasets.pbmc500()

    data = snap.pp.import_data(
        fragment_file,
        genome=snap.genome.hg38,
        file=h5ad(tmp_path),
        sorted_by_barcode=False,
    )
    snap.pp.add_tile_matrix(data)

    snap.pp.filter_cells(data)
    snap.pp.select_features(data)
    snap.pp.scrublet(data)
    snap.pp.call_doublets(data)

    snap.tl.spectral(data, sample_size = 100)
    snap.tl.spectral(data)
    snap.pp.knn(data)
    snap.tl.leiden(data)
    snap.tl.umap(data)

    snap.pp.make_gene_matrix(data, gff_file=snap.genome.hg38, file = h5ad(tmp_path), use_x = True)
    snap.pp.make_gene_matrix(data, gff_file=snap.genome.hg38, file = h5ad(tmp_path))