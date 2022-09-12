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

Transcriptional regulatory network (beta)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.init_network_from_annotation
   tl.add_cor_scores
   tl.add_regr_scores
   tl.prune_network

Plotting: `pl`
--------------

.. autosummary::
    :toctree: _autosummary

    pl.tsse
    pl.spectral_eigenvalues
    pl.scrublet
    pl.umap

Exporting: `ex`
---------------

.. autosummary::
    :toctree: _autosummary

    ex.export_bed
    ex.export_bigwig