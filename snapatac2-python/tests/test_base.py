import snapatac2 as snap

data = snap.pp.import_data(
    "data/fragments.bed.gz",
    "data/genes.gff3.gz",
    snap.genome.hg38
)

print("Make bin matrix")
snap.pp.make_tile_matrix(data)

print(data.X[...])