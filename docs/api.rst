.. automodule:: snapatac2

API
===

Import snapatac2 as::

    import snapatac2 as snap

Data IO
-------

(See more at `anndata-docs <https://anndata.readthedocs.io/en/latest/anndata.AnnData.html>`_)

.. autosummary::
    :toctree: _autosummary

    read
    read_mtx
    read_dataset
    create_dataset

Preprocessing: `pp`
-------------------

Basic Preprocessing
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    pp.import_data
    pp.make_tile_matrix
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

   tl.aggregate_X

Plotting: `pl`
--------------

.. autosummary::
    :toctree: _autosummary

    pl.tsse
    pl.spectral_eigenvalues
    pl.scrublet
    pl.umap

Exporting: `ex`
--------------

.. autosummary::
    :toctree: _autosummary

    ex.export_bed