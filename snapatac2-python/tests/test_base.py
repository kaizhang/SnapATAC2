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

print("selecting features...")
snap.pp.select_features(data)

print("scrublet...")
snap.pp.scrublet(data)
#snap.pp.call_doublets(data)
print(data)

print("spectral embedding...")
snap.tl.spectral(data)
snap.tl.spectral(data, sample_size = 300)

print("KNN...")
snap.pp.knn(data)

print("leiden...")
snap.tl.leiden(data)

print("umap...")
snap.tl.umap(data)

print("make gene matrix...")
snap.pp.make_gene_matrix(data, gff_file="data/genes.gff3.gz")
