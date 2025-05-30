===========
Tools: `tl`
===========
.. currentmodule:: snapatac2

Any transformation of the data matrix that is not *preprocessing*.
In contrast to a *preprocessing* function, a *tool* usually adds an easily
interpretable annotation to the data matrix, which can then be visualized with
a corresponding plotting function.

Embeddings
~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.spectral
   tl.multi_spectral
   tl.umap

Clustering
~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.leiden
   tl.kmeans
   tl.dbscan
   tl.hdbscan

Peak calling
~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.macs3
   tl.merge_peaks

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
   tl.prune_network

Utilities
~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.aggregate_X
   tl.aggregate_cells