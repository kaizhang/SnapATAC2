import snapatac2 as snap

print("import data...")
data = snap.pp.import_data(
    "data/fragments.bed.gz",
    "data/genes.gff3.gz",
    snap.genome.hg38
)
print(data)

print("make bin matrix...")
snap.pp.make_tile_matrix(data)
print(data)

print("plot tsse...")
snap.pl.tsse(data, out_file="out.png")

print("filtering...")
snap.pp.filter_cells(data, min_tsse = 15)
print(data)
