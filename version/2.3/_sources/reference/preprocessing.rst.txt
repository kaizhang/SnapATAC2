===================
Preprocessing: `pp`
===================
.. currentmodule:: snapatac2

BAM/Fragment file processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    pp.make_fragment_file
    pp.import_data

Additional Quality Control
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    pp.add_frip

Matrix operation
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

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
    pp.filter_doublets

Data Integration
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary
    
    pp.mnc_correct
    pp.harmony