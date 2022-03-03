from types import MappingProxyType
import h5py

from anndata._core.index import _subset, Index
from anndata._io.specs.registry import read_elem, write_elem, _REGISTRY, IOSpec
from anndata._core.sparse_dataset import SparseDataset, backed_csr_matrix
from anndata._core.views import as_view
from anndata.compat import (
    ZarrArray,
    ZarrGroup,
)


H5Array = h5py.Dataset
H5Group = h5py.Group

class InsertionCount(backed_csr_matrix):
    def __init__(self, group: h5py.Group):
        shape = tuple(group.attrs["shape"])
        dtype = group["data"].dtype
        super().__init__(shape, dtype=dtype)
        self.data = group["data"][:]
        self.indices = group["indices"][:]
        self.indptr = group["indptr"][:]
        self.reference_seq_name = group["reference_seq_name"][:]
        self.reference_seq_length = group["reference_seq_length"][:]

class InsertionCountView:
    def __init__(
        self,
        parent_mapping: InsertionCount,
        subset_idx: Index,
    ):
        self.parent_mapping = parent_mapping
        self.subset_idx = subset_idx

@as_view.register(InsertionCountView)
def as_view_insertion(d, view_args):
    return d

@_subset.register(InsertionCount)
def subset_insertion(d, subset_idx):
    return InsertionCountView(d, subset_idx)

@_REGISTRY.register_write(H5Group, InsertionCount, IOSpec("base_count", "0.1.0"))
@_REGISTRY.register_write(ZarrGroup, InsertionCount, IOSpec("base_count", "0.1.0"))
def write_sparse_dataset(f, k, elem, dataset_kwargs=MappingProxyType({})):
    g = f.create_group(k)
    g.attrs["shape"] = elem.shape
    g.attrs["h5sparse_format"] = "csr"

    g.create_dataset("data", data=elem.data, **dataset_kwargs)
    g.create_dataset("indices", data=elem.indices, **dataset_kwargs)
    g.create_dataset("indptr", data=elem.indptr, **dataset_kwargs)
    g.create_dataset("reference_seq_name", data=elem.reference_seq_name, **dataset_kwargs)
    g.create_dataset("reference_seq_length", data=elem.reference_seq_length, **dataset_kwargs)

@_REGISTRY.register_read(H5Group, IOSpec("base_count", "0.1.0"))
@_REGISTRY.register_read(ZarrGroup, IOSpec("base_count", "0.1.0"))
def read_insertion(elem):
    return InsertionCount(elem)


@_REGISTRY.register_read_partial(H5Group, IOSpec("base_count", "0.1.0"))
def read_sparse_partial(elem, *, items=None, indices=(slice(None), slice(None))):
    return InsertionCount(elem)[indices]

@_REGISTRY.register_write(H5Group, InsertionCountView, IOSpec("base_count", "0.1.0"))
@_REGISTRY.register_write(ZarrGroup, InsertionCountView, IOSpec("base_count", "0.1.0"))
def write_sparse_dataset_view(f, k, elem, dataset_kwargs=MappingProxyType({})):
    parent = elem.parent_mapping
    elem = parent[elem.subset_idx]

    g = f.create_group(k)
    g.attrs["shape"] = elem.shape
    g.attrs["h5sparse_format"] = "csr"

    g.create_dataset("data", data=elem.data, **dataset_kwargs)
    g.create_dataset("indices", data=elem.indices, **dataset_kwargs)
    g.create_dataset("indptr", data=elem.indptr, **dataset_kwargs)
    g.create_dataset("reference_seq_name", data=parent.reference_seq_name, **dataset_kwargs)
    g.create_dataset("reference_seq_length", data=parent.reference_seq_length, **dataset_kwargs)