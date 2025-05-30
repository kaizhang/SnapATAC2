========
Datasets
========
.. currentmodule:: snapatac2

These functions facilitate the download of public datasets and auxiliary data used
in the SnapATAC2 package.

.. note::

    You can change the data cache directory by setting the `SNAP_DATA_DIR` environmental
    variable.

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
    datasets.Meuleman_2020

Raw data
~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    datasets.pbmc500
    datasets.pbmc5k
    datasets.pbmc10k_multiome
    datasets.colon
    datasets.cre_HEA