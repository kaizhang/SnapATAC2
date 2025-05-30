import snapatac2 as snap
data = snap.read(snap.datasets.pbmc5k(type='h5ad'))
fig = snap.pl.tsse(data, show=False, out_file=None)
fig.show()
