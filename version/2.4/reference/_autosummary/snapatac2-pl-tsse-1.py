import snapatac2 as snap
data = snap.read(str(snap.datasets.pbmc5k(type='gene')))
fig = snap.pl.tsse(data, show=False, out_file=None)
fig.show()
