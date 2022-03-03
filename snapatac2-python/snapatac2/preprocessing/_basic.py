import numpy as np
import anndata as ad
import math
from typing import Optional, Union, Literal, Mapping
from anndata.experimental import AnnCollection

import snapatac2._snapatac2 as internal

def import_data(
    fragment_file: str,
    gff_file: str,
    chrom_size: Mapping[str, int],
    output: str = "data.h5ad",
    min_num_fragments: int = 200,
    min_tsse: float = 1,
    sorted_by_barcode: bool = True,
    backed: Optional[Literal["r", "r+"]] = "r+",
    n_jobs: int = 4,
) -> ad.AnnData:
    """
    Import dataset and compute QC metrics.

    Parameters
    ----------
    fragment_file
        File name of the fragment file
    gff_file
        File name of the gene annotation file in GFF format
    chrom_size
        A dictionary containing chromosome sizes, for example,
        `{"chr1": 2393, "chr2": 2344, ...}`
    output
        File name of the output h5ad file used to store the result
    min_num_fragments
        Threshold used to filter cells
    min_tsse
        Threshold used to filter cells
    sorted_by_barcode
        Whether the fragment file has been sorted by cell barcodes. Pre-sort the
        fragment file will speed up the processing and require far less memory.
    backed
        Whether to open the file in backed mode
    n_jobs
        number of CPUs to use
    
    Returns
    -------
    AnnData
    """
    internal.import_fragments(
        output, fragment_file, gff_file, chrom_size,
        min_num_fragments, min_tsse, sorted_by_barcode, n_jobs
    )
    return ad.read(output, backed=backed) 

def make_tile_matrix(
    adata: ad.AnnData,
    bin_size: int = 500,
    n_jobs: int = 4
):
    """
    Generate cell by bin count matrix.

    Parameters
    ----------
    adata
        AnnData
    bin_size
        The size of consecutive genomic regions used to record the counts
    n_jobs
        number of CPUs to use
    
    Returns
    -------
    """
    if not adata.isbacked:
        raise NotImplementedError("not implemented")
    filename = str(adata.filename)
    adata.file.close()
    internal.mk_tile_matrix(filename, bin_size, n_jobs)
    new_adata = ad.read(filename, backed="r+")
    new_adata.file.close()
    adata._init_as_actual(
        obs=new_adata.obs,
        var=new_adata.var,
        uns=new_adata.uns,
        obsm=new_adata.obsm,
        varm=new_adata.varm,
        varp=new_adata.varp,
        obsp=new_adata.obsp,
        raw=new_adata.raw,
        layers=new_adata.layers,
        shape=new_adata.shape,
        filename=new_adata.filename,
        filemode="r+",
    )

def make_gene_matrix(
    adata: Union[ad.AnnData, str],
    gff_file: str,
    output: Optional[str] = "gene_matrix.h5ad",
    backed: Optional[Literal["r", "r+"]] = None,
    n_jobs: int = 4,
) -> ad.AnnData:
    """
    Generate cell by gene activity matrix.

    Parameters
    ----------
    adata
        An anndata instance or the file name of a h5ad file containing the
        cell by bin count matrix
    gff_file
        File name of the gene annotation file in GFF format
    output
        File name of the h5ad file used to store the result
    backed
        Whether to open the file in backed mode
    n_jobs
        number of CPUs to use
    
    Returns
    -------
    AnnData
    """
    if isinstance(adata, ad.AnnData) and adata.isbacked:
        input_file = str(adata.filename)
        adata.file.close()
    elif isinstance(adata, str):
        input_file = adata
    else:
        raise NameError("Input type should be 'str' or 'AnnData'")
    internal.mk_gene_matrix(input_file, gff_file, output, n_jobs)
    gene_mat = ad.read(output, backed=backed)
    if backed is None:
        gene_mat._init_as_actual(
            X = gene_mat.X,
            obs=adata.obs,
            var=gene_mat.var,
            dtype=np.int32,
        )
    else:
        gene_mat.file.close()
        gene_mat._init_as_actual(
            obs=adata.obs,
            var=gene_mat.var,
            filename=output,
            filemode=backed,
        )
    return gene_mat

def filter_cells(
    data: ad.AnnData,
    min_counts: Optional[int] = 1000,
    min_tsse: Optional[float] = 5.0,
    max_counts: Optional[int] = None,
    max_tsse: Optional[float] = None,
    inplace: bool = True,
) -> Optional[np.ndarray]:
    """
    Filter cell outliers based on counts and numbers of genes expressed.
    For instance, only keep cells with at least `min_counts` counts or
    `min_tsse` TSS enrichment scores. This is to filter measurement outliers,
    i.e. "unreliable" observations.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_tsse
        Minimum TSS enrichemnt score required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_tsse
        Maximum TSS enrichment score expressed required for a cell to pass filtering.
    inplace
        Perform computation inplace or return result.

    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix:
    cells_subset
        Boolean index mask that does filtering. `True` means that the
        cell is kept. `False` means the cell is removed.
    """
    selected_cells = True
    if min_counts: selected_cells &= data.obs["n_fragment"] >= min_counts
    if max_counts: selected_cells &= data.obs["n_fragment"] <= max_counts
    if min_tsse: selected_cells &= data.obs["tsse"] >= min_tsse
    if max_tsse: selected_cells &= data.obs["tsse"] <= max_tsse

    if inplace:
        if data.isbacked:
            filename = str(data.filename)
            data[selected_cells, :].write()
            data.file.close()
            sdata = ad.read(filename, backed="r+")
            sdata.file.close()
            data._init_as_actual(
                obs=sdata.obs,
                var=sdata.var,
                uns=sdata.uns,
                obsm=sdata.obsm,
                varm=sdata.varm,
                varp=sdata.varp,
                obsp=sdata.obsp,
                raw=sdata.raw,
                layers=sdata.layers,
                shape=sdata.shape,
                filename=sdata.filename,
                filemode="r+",
            )
        else:
            data._inplace_subset_obs(selected_cells)
    else:
        return data[selected_cells, :]
 
def select_features(
    adata: Union[ad.AnnData, AnnCollection],
    variable_feature: bool = True,
    whitelist: Optional[str] = None,
    blacklist: Optional[str] = None,
    inplace: bool = True,
) -> Optional[np.ndarray]:
    """
    Perform feature selection.

    Parameters
    ----------
    adata
        AnnData object
    variable_feature
        Whether to perform feature selection using most variable features
    whitelist
        A user provided bed file containing genome-wide whitelist regions.
        Features that are overlapped with these regions will be retained.
    blacklist 
        A user provided bed file containing genome-wide blacklist regions.
        Features that are overlapped with these regions will be removed.
    inplace
        Perform computation inplace or return result.
    
    Returns
    -------
    Boolean index mask that does filtering. True means that the cell is kept.
    False means the cell is removed.
    """
    if isinstance(adata, ad.AnnData):
        count = np.ravel(adata.X[...].sum(axis = 0))
    else:
        count = np.zeros(adata.shape[1])
        for batch, _ in adata.iterate_axis(5000):
            count += np.ravel(batch.X[...].sum(axis = 0))

    selected_features = count != 0

    if whitelist is not None:
        selected_features &= internal.intersect_bed(list(adata.var_names), whitelist)
    if blacklist is not None:
        selected_features &= not internal.intersect_bed(list(adata.var_names), blacklist)

    if variable_feature:
        mean = count[selected_features].mean()
        std = math.sqrt(count[selected_features].var())
        selected_features &= np.absolute((count - mean) / std) < 1.65

    if inplace:
        adata.var["selected"] = selected_features
    else:
        return selected_features