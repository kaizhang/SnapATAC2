.. automodule:: snapatac2

API
===

Import snapatac2 as::

    import snapatac2 as snap

Data IO
-------

SnapATAC2 uses its own implementation of the AnnData format, for more details see
`anndata-rs-docs <http://kzhang.org/anndata-rs/api.html>`_.

.. autosummary::
    :toctree: _autosummary

    read
    read_mtx
    read_10x_mtx
    read_dataset
    create_dataset

Preprocessing: `pp`
-------------------

Basic Preprocessing
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    pp.make_fragment_file
    pp.import_data
    pp.add_tile_matrix
    pp.make_peak_matrix
    pp.make_gene_matrix
    pp.filter_cells
    pp.select_features
    pp.knn

Doublet removal
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    pp.scrublet
    pp.call_doublets

Data Integration
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary
    
    pp.mnc_correct
    pp.harmony

Tools: `tl`
-----------

Any transformation of the data matrix that is not *preprocessing*.
In contrast to a *preprocessing* function, a *tool* usually adds an easily
interpretable annotation to the data matrix, which can then be visualized with
a corresponding plotting function.

Embeddings
~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.spectral
   tl.laplacian
   tl.umap

Clustering
~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.leiden
   tl.kmeans
   tl.dbscan
   tl.hdbscan

Post-processing
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.call_peaks
   tl.aggregate_X

Differential analysis
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.marker_regions
   tl.diff_test

Motif analysis
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.motif_enrichment

Network analysis (beta)
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.init_network_from_annotation
   tl.add_cor_scores
   tl.add_regr_scores
   tl.add_tf_binding
   tl.link_tf_to_gene
   tl.to_gene_network
   tl.prune_network

Plotting: `pl`
--------------

.. autosummary::
    :toctree: _autosummary

    pl.tsse
    pl.spectral_eigenvalues
    pl.scrublet
    pl.umap
    pl.motif_enrichment
    pl.regions
    pl.network_edge_stat
    pl.render_plot

Exporting: `ex`
---------------

.. autosummary::
    :toctree: _autosummary

    ex.export_bed
    ex.export_bigwig

Datasets
--------

Genome
~~~~~~

.. autosummary::
    :toctree: _autosummary

    genome.Genome
    genome.GRCh37
    genome.GRCh38
    genome.GRCm38
    genome.GRCm39
    genome.hg19
    genome.hg38
    genome.mm10
    genome.mm39

Motifs
~~~~~~

.. autosummary::
    :toctree: _autosummary

    datasets.cis_bp

Raw data
~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    datasets.pbmc5k
    datasets.pbmc_multiome
    datasets.colon
    datasets.cre_HEA