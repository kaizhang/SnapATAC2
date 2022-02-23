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
    read_h5ad
    read_mtx
    read_10x_mtx

Preprocessing: `pp`
-------------------

Basic Preprocessing
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    pp.make_tile_matrix
    pp.make_gene_matrix
    pp.filter_cells
    pp.select_features
    pp.knn

Doublet removal
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    pp.scrublet

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

Plotting: `pl`
--------------

Quality Control
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    pl.tsse
    pl.spectral_eigenvalues