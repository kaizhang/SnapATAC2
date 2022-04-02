import snapatac2._snapatac2 as internal
from scipy.sparse import spmatrix
import pandas as pd
import polars
import numpy as np
from typing import Optional, Union

class AnnData:
    def __init__(
        self,
        *,
        filename: str = None,
        X = None,
        n_obs: int = None,
        n_vars: int = None,
        obs = None,
        var = None,
        obsm = None,
        pyanndata = None,
    ):
        if pyanndata is None:
            if X is not None: (n_obs, n_vars) = X.shape
            self._anndata = internal.PyAnnData(filename, n_obs, n_vars)
            if X is not None: self.X = X
            if obs is not None: self.obs = obs
            if var is not None: self.var = var
            if obsm is not None: self.obsm = obsm
        else:
            self._anndata = pyanndata

    @property
    def n_obs(self): return self._anndata.n_obs

    @property
    def n_vars(self): return self._anndata.n_vars

    @property
    def var_names(self): return self.var[:, 0].to_numpy()

    @property
    def obs_names(self): return self.obs[:, 0].to_numpy()

    @property
    def shape(self): return (self.n_obs, self.n_vars)

    @property
    def X(self): return self._anndata.get_x()

    @X.setter
    def X(self, X):
        self._anndata.set_x(X)
        if isinstance(X, spmatrix):
            ...
        else:
            ...

    @property
    def obs(self): 
        return self._anndata.get_obs()

    @obs.setter
    def obs(self, df):
        if isinstance(df, pd.DataFrame):
            df = polars.from_pandas(df)
        elif isinstance(df, dict):
            df = polars.from_dict(df)
        self._anndata.set_obs(df)

    @property
    def var(self):
        return self._anndata.get_var()

    @var.setter
    def var(self, df):
        if isinstance(df, pd.DataFrame):
            df = polars.from_pandas(df)
        elif isinstance(df, dict):
            df = polars.from_dict(df)
        self._anndata.set_var(df)

    @property
    def obsm(self):
        return OBSM(self._anndata)

    @obsm.setter
    def obsm(self, obsm):
        self._anndata.set_obsm(obsm)

    @property
    def obsp(self):
        return OBSP(self._anndata)

    @obsp.setter
    def obsp(self, obsp):
        self._anndata.set_obsp(obsp)

    @property
    def varm(self):
        return VARM(self._anndata)

    @varm.setter
    def varm(self, varm):
        self._anndata.set_varm(varm)

    @property
    def varp(self):
        return VARP(self._anndata)

    @varp.setter
    def varp(self, varp):
        self._anndata.set_varp(varp)

    @property
    def uns(self):
        return UNS(self._anndata)

    @uns.setter
    def uns(self, uns):
        self._anndata.set_uns(uns)

    def subset(self, obs_indices = None, var_indices = None):
        def to_indices(x, n):
            ifnone = lambda a, b: b if a is None else a
            if isinstance(x, slice):
                if x.stop is None:
                    pass
                    # do something with itertools.count()
                else:
                    return list(range(ifnone(x.start, 0), x.stop, ifnone(x.step, 1)))
            elif isinstance(x, np.ndarray) and x.ndim == 1 and x.size == n and x.dtype == bool:
                return list(x.nonzero()[0])
            elif isinstance(x, list):
                return x
            elif isinstance(x, np.ndarray):
                return list(x)
            else:
                return None

        i = to_indices(obs_indices, self.n_obs)
        j = to_indices(var_indices, self.n_vars)
        if i is None:
            if j is None:
                raise NameError("obs_indices and var_indices cannot be both None")
            else:
                self._anndata.subset_cols(j)
        else:
            if j is None:
                self._anndata.subset_rows(i)
            else:
                self._anndata.subset(i, j)

    def __repr__(self) -> str:
        descr = f"AnnData object with n_obs x n_vars = {self.n_obs} x {self.n_vars}"
        if self.obs is not None: descr += f"\n    obs: {str(self.obs[...].columns)[1:-1]}"
        if self.var is not None: descr += f"\n    var: {str(self.var[...].columns)[1:-1]}"
        for attr in [
            "obsm",
            "obsp",
            "varm",
            "varp",
            "uns",
        ]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def __str__(self) -> str:
        return self.__repr__()

    def write(self, filename: str):
        self._anndata.write(filename)

class OBSM:
    def __init__(self, anndata):
        self._anndata = anndata

    def __getitem__(self, key):
        return self._anndata.get_obsm(key)

    def __setitem__(self, key, data):
        self._anndata.add_obsm(key, data)

    def keys(self):
        return self._anndata.list_obsm()

    def __repr__(self) -> str:
        return f"AxisArrays with keys:\n{str(self.keys())[1:-1]}" 

class OBSP:
    def __init__(self, anndata):
        self._anndata = anndata

    def __getitem__(self, key):
        return self._anndata.get_obsp(key)

    def __setitem__(self, key, data):
        self._anndata.add_obsp(key, data)

    def keys(self):
        return self._anndata.list_obsp()

    def __repr__(self) -> str:
        return f"AxisArrays with keys: {self.keys()[1:-1]}" 

class VARM:
    def __init__(self, anndata):
        self._anndata = anndata

    def __getitem__(self, key):
        return self._anndata.get_varm(key)

    def __setitem__(self, key, data):
        self._anndata.add_varm(key, data)

    def keys(self):
        return self._anndata.list_varm()

    def __repr__(self) -> str:
        return f"AxisArrays with keys: {str(self.keys())[1:-1]}" 

class VARP:
    def __init__(self, anndata):
        self._anndata = anndata

    def __getitem__(self, key):
        return self._anndata.get_varp(key)

    def __setitem__(self, key, data):
        self._anndata.add_varp(key, data)

    def keys(self):
        return self._anndata.list_varp()

    def __repr__(self) -> str:
        return f"AxisArrays with keys: {str(self.keys())[1:-1]}" 

class UNS:
    def __init__(self, anndata):
        self._anndata = anndata

    def __getitem__(self, key):
        return self._anndata.get_uns(key)

    def __setitem__(self, key, data):
        self._anndata.add_uns(key, data)

    def keys(self):
        return self._anndata.list_uns()

    def __repr__(self) -> str:
        return f"Dict with keys: {str(self.keys())[1:-1]}" 

def read_h5ad(filename: str, mode: str = "r+") -> AnnData:
    return AnnData(pyanndata=internal.read_anndata(filename, mode))