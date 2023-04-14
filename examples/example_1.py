import snapatac2 as snap
import logging
print(snap.__version__)

fragment_file = snap.datasets.pbmc500()

print("######################################################")
logging.info("Import data - begin")
data = snap.pp.import_data(
    fragment_file,
    genome=snap.genome.hg38,
    gene_anno="/home/kaizhang/data/genome/GRCh38/gencode.v33.basic.annotation.gtf",
    file="pbmc.h5ad",
    sorted_by_barcode=False,
    low_memory=False,
)
print(data)
logging.info("Import data - OK")
print("######################################################\n")
data.close()

print("######################################################")
logging.info("Import data (low memory) - begin")
data = snap.pp.import_data(
    fragment_file,
    genome=snap.genome.hg38,
    file="pbmc.h5ad",
    sorted_by_barcode=False,
    low_memory=True,
)
print(data)
logging.info("Import data (low memory) - OK")
print("######################################################\n")

print("######################################################")
logging.info("make bin matrix - begin")
snap.pp.add_tile_matrix(data)
print(data)
logging.info("make bin matrix - OK")
print("######################################################\n")

#snap.pl.tsse(data, out_file="tsse.png")

print("######################################################")
logging.info("Filter cells - begin")
snap.pp.filter_cells(data)
print(data)
logging.info("Filter cells - OK")
print("######################################################\n")

print("######################################################")
logging.info("select features - begin")
snap.pp.select_features(data)
logging.info("select features - OK")
print("######################################################\n")

print("######################################################")
logging.info("scrublet - begin")
snap.pp.scrublet(data)
snap.pp.filter_doublets(data)
print(data)
logging.info("scrublet - OK")
print("######################################################\n")

print("######################################################")
logging.info("spectral embedding - begin")
snap.tl.spectral(data, sample_size = 100)
snap.tl.spectral(data)
logging.info("spectral embedding - OK")
print("######################################################\n")

print("######################################################")
logging.info("KNN - begin")
snap.pp.knn(data)
logging.info("KNN - OK")
print("######################################################\n")

print("######################################################")
logging.info("leiden - begin")
snap.tl.leiden(data)
logging.info("leiden - OK")
print("######################################################\n")

print("######################################################")
logging.info("UMAP - begin")
snap.tl.umap(data)
logging.info("UMAP - OK")
print("######################################################\n")

print("######################################################")
logging.info("call peaks - begin")
snap.tl.call_peaks(data, groupby="leiden")
logging.info("call peaks - OK")
print("######################################################\n")

print("######################################################")
logging.info("export bigwig files - begin")
snap.ex.export_bigwig(data, groupby="leiden")
logging.info("export bigwig files - OK")
print("######################################################\n")

print("######################################################")
logging.info("make peak matrix - begin")
snap.pp.make_peak_matrix(data, file="peak_matrix.h5ad")
logging.info("make peak matrix - OK")
print("######################################################\n")

print("######################################################")
logging.info("make gene matrix - begin")
snap.pp.make_gene_matrix(
    data, 
    gene_anno="/home/kaizhang/data/genome/GRCh38/gencode.v33.basic.annotation.gtf",
    file = "gene_matrix.h5ad",
    use_x = True
)
snap.pp.make_gene_matrix(
    data,
    gene_anno=snap.genome.hg38,
    file = "gene_matrix.h5ad"
)
logging.info("make gene matrix - OK")
print("######################################################\n")
